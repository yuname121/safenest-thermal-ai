#pragma once

// Copy this file to wifi_secrets.h and replace every placeholder locally.
// wifi_secrets.h is ignored by Git. Do not commit Wi-Fi credentials or a
// private Raspberry Pi address.

#define THERMAL_WIFI_SSID "REPLACE_WITH_2G_WIFI_SSID"
#define THERMAL_WIFI_PASSWORD "REPLACE_WITH_WIFI_PASSWORD"

// Set this to the Raspberry Pi WLAN IPv4 address shown by: hostname -I
#define THERMAL_RECEIVER_IP "192.168.0.100"
#define THERMAL_RECEIVER_PORT 5005
