/*
 * Thermal-90 UDP raw-frame sender for Seeed XIAO ESP32-C6.
 *
 * Protocol compatibility:
 *   - One logical frame is exactly 10,080 bytes.
 *   - MTU-safe UDP chunks carry an explicit frame/chunk envelope and CRC32.
 *   - The reassembled payload is 5,040 uint16 words in ESP32 little-endian
 *     memory order, unchanged from the Thermal_Test frame.
 *   - Words 0..79 are the unmodified sensor header.
 *   - Words 80..5039 are the unmodified 80 x 62 pixel payload.
 *
 * This sketch intentionally sends no normalized image, colour map, model input,
 * or temperature conversion. The Raspberry Pi collector preserves the exact
 * datagram and records the decoding metadata separately.
 */

#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

#include "wifi_secrets.h"

// Pin map copied from the working Thermal_Test prototype for XIAO ESP32-C6.
#define PIN_SDA D4
#define PIN_SCL D5
#define PIN_MOSI D10
#define PIN_MISO D9
#define PIN_CLK D8
#define PIN_CS D3
#define PIN_D_READY D1
#define PIN_NRESET D2

// Thermal-90 / MI48xx prototype transport contract.
constexpr uint8_t THERMAL_I2C_ADDRESS = 0x40;
constexpr uint32_t THERMAL_SPI_SPEED_HZ = 2000000;
constexpr uint16_t THERMAL_HEADER_WORDS = 80;
constexpr uint16_t THERMAL_WIDTH = 80;
constexpr uint16_t THERMAL_HEIGHT = 62;
constexpr uint16_t THERMAL_PIXEL_WORDS = THERMAL_WIDTH * THERMAL_HEIGHT;
constexpr uint16_t THERMAL_FRAME_WORDS = THERMAL_HEADER_WORDS + THERMAL_PIXEL_WORDS;
constexpr size_t THERMAL_FRAME_BYTES = THERMAL_FRAME_WORDS * sizeof(uint16_t);
constexpr uint16_t THERMAL_UDP_LOCAL_PORT = 40000;

// SafeNest Thermal raw UDP V2. Header integers use network byte order. The
// payload remains the original little-endian 10,080-byte sensor frame.
constexpr char THERMAL_UDP_MAGIC[] = "SNTR";
constexpr uint8_t THERMAL_UDP_VERSION = 2;
constexpr uint8_t THERMAL_UDP_MESSAGE_TYPE_RAW_U16_LE = 1;
constexpr size_t THERMAL_UDP_HEADER_BYTES = 32;
constexpr size_t THERMAL_UDP_DATAGRAM_BYTES = 1200;
constexpr size_t THERMAL_UDP_CHUNK_BYTES =
    THERMAL_UDP_DATAGRAM_BYTES - THERMAL_UDP_HEADER_BYTES;
constexpr uint16_t THERMAL_UDP_CHUNK_COUNT =
    (THERMAL_FRAME_BYTES + THERMAL_UDP_CHUNK_BYTES - 1) /
    THERMAL_UDP_CHUNK_BYTES;

static_assert(THERMAL_FRAME_WORDS == 5040, "Thermal_Test protocol requires 5,040 words");
static_assert(THERMAL_FRAME_BYTES == 10080, "Thermal_Test protocol requires 10,080 bytes");
static_assert(THERMAL_UDP_CHUNK_COUNT == 9, "Raw UDP V2 requires nine chunks per frame");

WiFiUDP udp;
IPAddress receiverIp;
uint16_t frameWords[THERMAL_FRAME_WORDS];
uint8_t thermalUdpDatagram[THERMAL_UDP_DATAGRAM_BYTES];

volatile uint32_t dataReadySignals = 0;
uint32_t sentFrames = 0;
uint32_t sendFailures = 0;
uint32_t droppedReadySignals = 0;
uint32_t transportFrameId = 0;
unsigned long lastWifiAttemptMs = 0;

void putU16(uint8_t* output, uint16_t value) {
  output[0] = static_cast<uint8_t>(value >> 8);
  output[1] = static_cast<uint8_t>(value);
}

void putU32(uint8_t* output, uint32_t value) {
  output[0] = static_cast<uint8_t>(value >> 24);
  output[1] = static_cast<uint8_t>(value >> 16);
  output[2] = static_cast<uint8_t>(value >> 8);
  output[3] = static_cast<uint8_t>(value);
}

uint32_t rawFrameCrc32(const uint8_t* data, size_t length) {
  uint32_t crc = 0xFFFFFFFFU;
  for (size_t offset = 0; offset < length; ++offset) {
    crc ^= data[offset];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320U & (0U - (crc & 1U)));
    }
  }
  return ~crc;
}

void IRAM_ATTR onDataReady() {
  dataReadySignals++;
}

bool parseReceiverIp() {
  if (!receiverIp.fromString(THERMAL_RECEIVER_IP)) {
    Serial.println("[FATAL] THERMAL_RECEIVER_IP is not a valid IPv4 address.");
    return false;
  }
  return true;
}

void ensureWifiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  const unsigned long now = millis();
  if (now - lastWifiAttemptMs < 5000) {
    return;
  }
  lastWifiAttemptMs = now;
  Serial.println("[WiFi] reconnecting...");
  WiFi.disconnect();
  WiFi.begin(THERMAL_WIFI_SSID, THERMAL_WIFI_PASSWORD);
}

bool waitForSensorBoot() {
  for (int attempt = 0; attempt < 50; ++attempt) {
    Wire.beginTransmission(THERMAL_I2C_ADDRESS);
    Wire.write(0xB6);  // status register
    if (Wire.endTransmission(false) != 0) {
      delay(100);
      continue;
    }
    if (Wire.requestFrom(static_cast<uint16_t>(THERMAL_I2C_ADDRESS), static_cast<uint8_t>(1)) == 1 &&
        Wire.available()) {
      const uint8_t status = Wire.read();
      if ((status & 0x20) == 0) {  // BOOTING_UP bit clear
        return true;
      }
    }
    delay(100);
  }
  return false;
}

bool configureSensor() {
  Wire.beginTransmission(THERMAL_I2C_ADDRESS);
  if (Wire.endTransmission() != 0) {
    Serial.println("[FATAL] Thermal sensor not found at I2C address 0x40.");
    return false;
  }

  // Power-up command, frame-rate register, and continuous-stream command are
  // retained from the verified Thermal_Test prototype. The register value is a
  // configuration request only; the Pi must measure effective FPS.
  Wire.beginTransmission(THERMAL_I2C_ADDRESS);
  Wire.write(0xB0);
  Wire.write(0x13);
  if (Wire.endTransmission() != 0 || !waitForSensorBoot()) {
    Serial.println("[FATAL] Thermal sensor did not finish booting.");
    return false;
  }

  Wire.beginTransmission(THERMAL_I2C_ADDRESS);
  Wire.write(0xB4);
  Wire.write(0x04);  // approximately 7 FPS in the prototype; not a measurement
  if (Wire.endTransmission() != 0) {
    Serial.println("[FATAL] Could not write the frame-rate register.");
    return false;
  }

  Wire.beginTransmission(THERMAL_I2C_ADDRESS);
  Wire.write(0xB1);
  Wire.write(0x02);  // continuous stream mode
  if (Wire.endTransmission() != 0) {
    Serial.println("[FATAL] Could not enable continuous stream mode.");
    return false;
  }

  return true;
}

bool takeFrame() {
  SPI.beginTransaction(SPISettings(THERMAL_SPI_SPEED_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW);
  delayMicroseconds(100);
  for (uint16_t index = 0; index < THERMAL_FRAME_WORDS; ++index) {
    frameWords[index] = SPI.transfer16(0x0000);
  }
  delayMicroseconds(100);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
  return true;
}

bool sendRawFrame(uint32_t frameId) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }
  const uint8_t* rawFrame = reinterpret_cast<const uint8_t*>(frameWords);
  const uint32_t frameCrc32 = rawFrameCrc32(rawFrame, THERMAL_FRAME_BYTES);
  for (uint16_t chunkIndex = 0; chunkIndex < THERMAL_UDP_CHUNK_COUNT; ++chunkIndex) {
    const uint32_t offset = chunkIndex * THERMAL_UDP_CHUNK_BYTES;
    const size_t remaining = THERMAL_FRAME_BYTES - offset;
    const uint16_t length = static_cast<uint16_t>(
        remaining < THERMAL_UDP_CHUNK_BYTES ? remaining : THERMAL_UDP_CHUNK_BYTES);
    uint8_t* header = thermalUdpDatagram;
    memcpy(header, THERMAL_UDP_MAGIC, 4);
    header[4] = THERMAL_UDP_VERSION;
    header[5] = THERMAL_UDP_MESSAGE_TYPE_RAW_U16_LE;
    putU16(header + 6, THERMAL_UDP_HEADER_BYTES);
    putU32(header + 8, frameId);
    putU16(header + 12, chunkIndex);
    putU16(header + 14, THERMAL_UDP_CHUNK_COUNT);
    putU32(header + 16, THERMAL_FRAME_BYTES);
    putU32(header + 20, offset);
    putU16(header + 24, length);
    putU16(header + 26, 0);
    putU32(header + 28, frameCrc32);
    memcpy(header + THERMAL_UDP_HEADER_BYTES, rawFrame + offset, length);
    const size_t datagramBytes = THERMAL_UDP_HEADER_BYTES + length;
    if (!udp.beginPacket(receiverIp, THERMAL_RECEIVER_PORT) ||
        udp.write(thermalUdpDatagram, datagramBytes) != datagramBytes ||
        udp.endPacket() != 1) {
      return false;
    }
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(250);
  Serial.println("\n[SafeNest Thermal-90 raw UDP sender]");

  if (!parseReceiverIp()) {
    while (true) delay(1000);
  }

  pinMode(PIN_CS, OUTPUT);
  pinMode(PIN_NRESET, OUTPUT);
  pinMode(PIN_D_READY, INPUT);
  digitalWrite(PIN_CS, HIGH);

  digitalWrite(PIN_NRESET, LOW);
  delay(10);
  digitalWrite(PIN_NRESET, HIGH);
  delay(100);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  if (!configureSensor()) {
    while (true) delay(1000);
  }

  SPI.begin(PIN_CLK, PIN_MISO, PIN_MOSI, PIN_CS);
  attachInterrupt(digitalPinToInterrupt(PIN_D_READY), onDataReady, RISING);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(THERMAL_WIFI_SSID, THERMAL_WIFI_PASSWORD);
  udp.begin(THERMAL_UDP_LOCAL_PORT);

  Serial.printf("[Protocol] SafeNest Thermal raw UDP V2: %u bytes/frame, %u chunks, %u x %u pixels\n",
                static_cast<unsigned>(THERMAL_FRAME_BYTES), THERMAL_UDP_CHUNK_COUNT,
                THERMAL_WIDTH, THERMAL_HEIGHT);
  Serial.printf("[Receiver] %s:%u\n", THERMAL_RECEIVER_IP, THERMAL_RECEIVER_PORT);
}

void loop() {
  ensureWifiConnected();

  bool frameReady = false;
  uint32_t queuedSignals = 0;
  noInterrupts();
  if (dataReadySignals > 0) {
    queuedSignals = dataReadySignals;
    dataReadySignals = 0;
    frameReady = true;
  }
  interrupts();

  if (!frameReady) {
    delay(1);
    return;
  }

  if (queuedSignals > 1) {
    droppedReadySignals += queuedSignals - 1;
  }

  if (!takeFrame()) {
    Serial.println("[ERROR] SPI frame acquisition failed.");
    return;
  }

  const uint16_t frameCounter = frameWords[0];
  const uint32_t currentTransportFrameId = transportFrameId++;
  if (sendRawFrame(currentTransportFrameId)) {
    sentFrames++;
  } else {
    sendFailures++;
  }

  if ((sentFrames + sendFailures) % 30 == 0) {
    Serial.printf("[Stats] transport_frame_id=%lu frame_counter=%u sent=%lu send_failures=%lu ready_drops=%lu wifi=%s\n",
                  static_cast<unsigned long>(currentTransportFrameId), frameCounter,
                  static_cast<unsigned long>(sentFrames),
                  static_cast<unsigned long>(sendFailures),
                  static_cast<unsigned long>(droppedReadySignals),
                  WiFi.status() == WL_CONNECTED ? "connected" : "disconnected");
  }
}
