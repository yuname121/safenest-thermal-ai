#!/usr/bin/env python3
"""Generate deterministic compact evidence for Thermal T-A0.

This generator records the 2026-08-10 local-first audit. It never opens a
macOS dataless payload and never extracts or reconstructs an archive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/thermal/manifests/T-A0_source_identity"
REPORT = ROOT / "docs/reports/20260810_Codex_T-A0_Thermal_Dataset_Selection_Source_Identity_01.md"
ACCESS_DATE = "2026-08-10"


def candidate(candidate_id: str, **updates: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "candidate_id": candidate_id,
        "official_dataset_name": "UNKNOWN",
        "stable_identifier": None,
        "official_distribution_location": None,
        "original_publication": None,
        "license_terms": "LICENSE_UNVERIFIED",
        "research_use_permission": "NOT_VERIFIABLE",
        "model_training_permission": "NOT_VERIFIABLE",
        "redistribution_restrictions": "NOT_VERIFIABLE",
        "access_registration_requirements": "NOT_VERIFIABLE",
        "genuine_thermal_status": "UNKNOWN",
        "rgb_colorized_only": "UNKNOWN",
        "representation_classification": "UNKNOWN",
        "sensor_model": "UNKNOWN",
        "wavelength": "UNKNOWN",
        "source_frame_shape": "UNKNOWN",
        "source_orientation": "UNKNOWN",
        "dtype": "UNKNOWN",
        "bit_depth": "UNKNOWN",
        "channels": "UNKNOWN",
        "file_format": "UNKNOWN",
        "frame_rate": "UNKNOWN",
        "timestamp_availability": "UNKNOWN",
        "subject_identifiers": "UNKNOWN",
        "session_identifiers": "UNKNOWN",
        "scene_identifiers": "UNKNOWN",
        "sequence_identifiers": "UNKNOWN",
        "event_identifiers": "UNKNOWN",
        "camera_identifiers": "UNKNOWN",
        "fall_labels": "UNKNOWN",
        "fall_event_boundary_quality": "UNKNOWN",
        "normal_activity_coverage": "UNKNOWN",
        "hard_negative_coverage": "UNKNOWN",
        "staged_vs_natural_fall_semantics": "UNKNOWN",
        "subject_count": "UNKNOWN",
        "session_count": "UNKNOWN",
        "sequence_count": "UNKNOWN",
        "event_count": "UNKNOWN",
        "subject_wise_split_feasibility": "NOT_VERIFIABLE",
        "fallback_grouping_feasibility": "NOT_VERIFIABLE",
        "duplicate_near_duplicate_risk": "UNKNOWN",
        "event_level_evaluation_compatibility": "NOT_VERIFIABLE",
        "approximate_download_storage_impact": "UNKNOWN",
        "checksum_availability": "UNKNOWN",
        "thermal44_relevance": "THERMAL44_COMPARISON_NOT_VERIFIABLE",
        "known_limitations": [],
        "materialization_state": "NOT_LOCAL",
        "overall_status": "NEEDS_MANUAL_REVIEW",
        "explicit_justification": "Insufficient evidence.",
        "source_identity_status": "SOURCE_IDENTITY_UNVERIFIED",
        "license_status": "LICENSE_UNVERIFIED",
        "inventory_status": "NOT_VERIFIABLE",
        "label_semantics_status": "LABEL_SEMANTICS_UNVERIFIED",
        "grouping_status": "GROUP_PROVENANCE_UNVERIFIED",
        "safe_reader_documentation_status": "NOT_VERIFIABLE",
        "official_source_or_limitation": "No verified official source is available.",
        "evidence_category": "UNKNOWN",
    }
    item.update(updates)
    return item


CANDIDATES = [
    candidate(
        "external_ehomeseniors_2019",
        official_dataset_name="eHomeSeniors Dataset: An Infrared Thermal Sensor Dataset for Automatic Fall Detection Research",
        stable_identifier="doi:10.3390/s19204565",
        official_distribution_location="https://www.mdpi.com/1424-8220/19/20/4565/s1",
        original_publication="https://doi.org/10.3390/s19204565",
        license_terms="ARTICLE_AND_PUBLISHER_SUPPLEMENT_PAGE_CC_BY_4_0; DATA_SPECIFIC_TERMS_NOT_SEPARATELY_STATED",
        research_use_permission="PERMITTED_BY_CC_BY_4_0_SUBJECT_TO_ATTRIBUTION; DATA_SPECIFIC_CONFIRMATION_RECOMMENDED",
        model_training_permission="PERMITTED_BY_CC_BY_4_0_INFERENCE; DATA_SPECIFIC_CONFIRMATION_RECOMMENDED",
        redistribution_restrictions="ATTRIBUTION_REQUIRED; DATA_SPECIFIC_CONFIRMATION_RECOMMENDED",
        access_registration_requirements="OPEN_PUBLISHER_SUPPLEMENT; NO_REGISTRATION_DOCUMENTED",
        genuine_thermal_status="REAL_THERMAL_NUMERIC_MEASUREMENTS",
        rgb_colorized_only=False,
        representation_classification="MULTIMODAL",
        representation_details={
            "melexis": "RADIOMETRIC_TEMPERATURE plus documented raw sensor fields",
            "omron": "RADIOMETRIC_TEMPERATURE",
        },
        sensor_model=["Melexis MLX90640", "Omron D6T-8L-06"],
        wavelength="FAR_INFRARED for MLX90640; exact wavelength not stated in the dataset paper",
        source_frame_shape={"melexis": [24, 32], "omron": "four 1x8 arrays recorded as 32 temperature values"},
        source_orientation="WALL_MOUNTED; MLX90640 horizontal FOV 37.5 degrees, vertical FOV 55 degrees",
        dtype="decimal text values in CSV; numeric MAT representation also supplied",
        bit_depth="NOT_APPLICABLE_TO_EXPORTED_TEMPERATURE_TEXT",
        channels=1,
        file_format=["CSV", "MAT"],
        frame_rate={"melexis_hz": "approximately 16", "omron_hz": "NOT_STATED_IN_REVIEWED_SOURCE"},
        timestamp_availability="ROW_LEVEL_COLLECTION_DATETIME",
        subject_identifiers="GX-Y in file names identifies one of six volunteers",
        session_identifiers="FILE_LEVEL fall-type recording; five falls are contained in most files",
        scene_identifiers="SINGLE_DOCUMENTED_LAB_ROOM",
        sequence_identifiers="FILE_NAME",
        event_identifiers="FALL_TYPE f01-f15; individual repeated-fall boundaries not explicitly annotated",
        camera_identifiers="SENSOR_NAME in file name",
        fall_labels="15 safely staged fall types; no natural real-world older-adult falls",
        fall_event_boundary_quality="FIVE_FALLS_PER_FILE_BUT_NO_REVIEWED_FRAMEWISE_ONSET_END_ANNOTATION",
        normal_activity_coverage="NONE_DOCUMENTED_IN_DISTRIBUTED 180 FALL FILES",
        hard_negative_coverage="NONE_DOCUMENTED",
        staged_vs_natural_fall_semantics="STAGED; three performing artists were coached by a physiotherapist and three healthy young volunteers were uncoached",
        subject_count=6,
        session_count="UNKNOWN; 180 recording files are documented but are not asserted to be independent sessions",
        sequence_count=180,
        event_count="898 file-contained fall repetitions across both sensor exports; paired physical-event count and boundaries are not verified",
        subject_wise_split_feasibility="YES_BY_GX_Y",
        fallback_grouping_feasibility="YES_BY_FILE_AND_FALL_TYPE",
        duplicate_near_duplicate_risk="HIGH_WITHIN_FILE_DUE_TO_ADJACENT_FRAMES; GROUP_BY_SUBJECT_AND_FILE",
        event_level_evaluation_compatibility="BLOCKED_WITHOUT_RECONSTRUCTED_EVENT_BOUNDARIES_AND_NORMAL_SEQUENCES",
        approximate_download_storage_impact="Publisher supplement shown as approximately 265.2 MB compressed",
        checksum_availability="NO_OFFICIAL_CHECKSUM FOUND ON REVIEWED PUBLISHER PAGE",
        known_limitations=[
            "No normal/ADL or hard-negative sequences are documented in the distributed fall files.",
            "Individual onset/end boundaries for the five falls within a file are not documented in the reviewed source.",
            "Dataset-specific license wording is not separated from the CC BY 4.0 article/supplement license.",
        ],
        overall_status="NEEDS_MANUAL_REVIEW",
        explicit_justification="Strong subject and temperature provenance make this a technical backup, but no normal activities and no explicit per-event boundaries prevent standalone fall-evaluation selection.",
        source_identity_status="VERIFIED",
        license_status="NEEDS_MANUAL_REVIEW",
        inventory_status="OFFICIAL_DOCUMENTATION_ONLY_NOT_DOWNLOADED",
        label_semantics_status="PARTIAL",
        grouping_status="SUBJECT_GROUPING_AVAILABLE",
        safe_reader_documentation_status="DOCUMENTED",
        official_source_or_limitation="Official MDPI article and publisher supplement reviewed.",
        evidence_category="OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
    ),
    candidate(
        "external_muvim_2022",
        official_dataset_name="Multi Visual Modality Fall Detection Dataset (MUVIM)",
        stable_identifier="doi:10.1109/ACCESS.2022.3211939",
        official_distribution_location="DATA_AVAILABLE_BY_AUTHOR_REQUEST_PER_PUBLICATION",
        original_publication="https://doi.org/10.1109/ACCESS.2022.3211939",
        license_terms="LICENSE_UNVERIFIED",
        research_use_permission="REQUEST_AND_APPROVAL_REQUIRED",
        model_training_permission="NOT_VERIFIABLE_UNTIL_TERMS_RECEIVED",
        redistribution_restrictions="NOT_VERIFIABLE",
        access_registration_requirements="AUTHOR_REQUEST_REQUIRED",
        genuine_thermal_status="REAL_THERMAL_VIDEO",
        rgb_colorized_only="VIDEO_ENCODING_NOT_VERIFIED_AS_RADIOMETRIC",
        representation_classification="THERMAL_COLORIZED_RENDERING",
        sensor_model="Three FLIR ONE Gen 3 smartphone thermal cameras",
        wavelength="UNKNOWN",
        source_frame_shape=[1080, 1440, 3],
        source_orientation="THREE CEILING-MOUNTED VIEWS (left, centre, right)",
        dtype="8-bit compressed video representation expected; exact decoded dtype not verified",
        bit_depth=8,
        channels=3,
        file_format="MP4",
        frame_rate=8.7,
        timestamp_availability="SYNCHRONIZED_TRIALS_DOCUMENTED; DISTRIBUTED TIMESTAMP SCHEMA NOT REVIEWABLE WITHOUT ACCESS",
        subject_identifiers="30 younger adults with falls/ADL plus 10 older adults with ADL",
        session_identifiers="DOCUMENTED",
        scene_identifiers="DESIGNED HOME ENVIRONMENT",
        sequence_identifiers="DOCUMENTED_CONTINUOUS_TRIALS",
        event_identifiers="DOCUMENTED_FALL_AND_ADL EVENTS; FILE SCHEMA ACCESS BLOCKED",
        camera_identifiers="LEFT_CENTRE_RIGHT THERMAL CAMERAS",
        fall_labels="STAGED FALLS AND CONTINUOUS ADL",
        fall_event_boundary_quality="PUBLICATION_DESCRIBES EVENTS; MACHINE-READABLE BOUNDARIES NOT VERIFIED",
        normal_activity_coverage="CONTINUOUS ADL INCLUDING TEN OLDER ADULTS",
        hard_negative_coverage="VARIED ADL, FURNITURE, LIGHTING AND CAMERA PLACEMENT",
        staged_vs_natural_fall_semantics="STAGED FALLS; no natural falls claimed",
        subject_count=40,
        session_count="UNKNOWN",
        sequence_count="UNKNOWN",
        event_count="UNKNOWN",
        subject_wise_split_feasibility="LIKELY, BUT DISTRIBUTED IDENTIFIERS NOT VERIFIED",
        fallback_grouping_feasibility="LIKELY_BY_TRIAL",
        duplicate_near_duplicate_risk="HIGH_ADJACENT_FRAME_RISK; GROUP_BY_SUBJECT_TRIAL_CAMERA",
        event_level_evaluation_compatibility="POTENTIALLY_COMPATIBLE_AFTER_ACCESS_AND SCHEMA AUDIT",
        approximate_download_storage_impact="UNKNOWN; multi-camera multimodal video likely large",
        checksum_availability="NOT_VERIFIABLE",
        known_limitations=["Access is by request.", "License and redistribution terms were not publicly verified.", "Thermal MP4 is not established as radiometric."],
        overall_status="ACCESS_BLOCKED",
        explicit_justification="Promising grouping and activity coverage, but access, license, file semantics and checksums cannot be verified locally.",
        source_identity_status="VERIFIED",
        license_status="LICENSE_UNVERIFIED",
        inventory_status="ACCESS_BLOCKED",
        label_semantics_status="PUBLICATION_ONLY",
        grouping_status="PUBLICATION_ONLY",
        safe_reader_documentation_status="ACCESS_BLOCKED",
        official_source_or_limitation="Original publication verified; dataset distribution requires author access.",
        evidence_category="OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
    ),
    candidate(
        "local_additional_human_not_human",
        official_dataset_name="UNKNOWN LOCAL human/not human annotated image tree",
        official_distribution_location=None,
        genuine_thermal_status="THERMAL_VISUALIZATION_SCREENSHOTS_OR_EXPORTS; ORIGINAL_NUMERIC_THERMAL NOT PRESENT",
        rgb_colorized_only=True,
        representation_classification="THERMAL_COLORIZED_RENDERING",
        source_frame_shape="MULTIPLE HIGH-RESOLUTION RGB/RGBA SHAPES",
        dtype="uint8",
        bit_depth=8,
        channels=[3, 4],
        file_format=["JPEG", "PNG", "JSON polygon annotations"],
        subject_identifiers="ABSENT",
        session_identifiers="ABSENT",
        sequence_identifiers="ABSENT",
        event_identifiers="ABSENT",
        camera_identifiers="ABSENT",
        fall_labels="ABSENT; labels are human detected / not human detected",
        fall_event_boundary_quality="NOT_APPLICABLE",
        normal_activity_coverage="NOT_LABELED",
        hard_negative_coverage="NOT_VERIFIABLE",
        staged_vs_natural_fall_semantics="NOT_APPLICABLE",
        subject_wise_split_feasibility="NO",
        fallback_grouping_feasibility="NO_AUTHORITATIVE_GROUP",
        duplicate_near_duplicate_risk="HIGH; screenshot-like captures and no sequence provenance",
        event_level_evaluation_compatibility="NO",
        approximate_download_storage_impact="98,992,778 logical bytes in current local namespace",
        checksum_availability="PARTIAL_LOCAL_CONTENT_AGGREGATE; 197 image placeholders excluded",
        known_limitations=["Presence labels are not fall labels.", "Original source, license, sensor and conversion history are unknown.", "197 of 410 images are cloud placeholders."],
        materialization_state="PARTIALLY_MATERIALIZED",
        overall_status="REJECTED_PROVENANCE",
        explicit_justification="The tree contains rendered high-resolution images and presence polygons, not source thermal fall events with grouping provenance.",
        source_identity_status="SOURCE_IDENTITY_UNVERIFIED",
        inventory_status="DETERMINISTIC_METADATA_INVENTORY",
        label_semantics_status="VERIFIED_PRESENCE_ONLY_UNUSABLE_FOR_FALL",
        grouping_status="GROUP_PROVENANCE_UNVERIFIED",
        safe_reader_documentation_status="NOT_VERIFIABLE",
        official_source_or_limitation="No repository clue or embedded origin identifies an official source.",
        evidence_category="LOCALLY_MEASURED",
    ),
    candidate(
        "local_family_a_fall_non_fall_png",
        official_dataset_name="UNKNOWN LOCAL Thermal_Dataset_Fall_Non_Fall collection",
        official_distribution_location=None,
        genuine_thermal_status="THERMAL_RENDERING_APPEARANCE_ONLY; ORIGINAL_THERMAL_VALUES NOT PRESENT",
        rgb_colorized_only=True,
        representation_classification="THERMAL_COLORIZED_RENDERING",
        source_frame_shape=[226, 230, 3],
        dtype="uint8",
        bit_depth=8,
        channels=3,
        file_format="PNG",
        subject_identifiers="UNKNOWN; filename prefixes must not be interpreted",
        session_identifiers="UNKNOWN",
        sequence_identifiers="UNKNOWN",
        event_identifiers="UNKNOWN",
        camera_identifiers="UNKNOWN",
        fall_labels="UNKNOWN; directory name is not file-level label evidence",
        fall_event_boundary_quality="ABSENT_OR_UNKNOWN",
        normal_activity_coverage="UNKNOWN",
        hard_negative_coverage="UNKNOWN",
        staged_vs_natural_fall_semantics="UNKNOWN",
        subject_wise_split_feasibility="NO_VERIFIED_SUBJECT_ID",
        fallback_grouping_feasibility="NO_AUTHORITATIVE_PREFIX_MEANING",
        duplicate_near_duplicate_risk="HIGH; flat adjacent numbered frames with unknown sequence boundaries",
        event_level_evaluation_compatibility="NO",
        approximate_download_storage_impact="224,906,370 logical bytes; 132,087,808 allocated bytes at audit time",
        checksum_availability="PARTIAL_LOCAL_CONTENT_AGGREGATE; 3,025 placeholders excluded",
        known_limitations=["ILS, SSJ, ILP and SP meanings remain unknown.", "RGB color mapping removed any verified physical unit.", "3,025 of 6,748 files are dataless placeholders.", "No official source or license was identified."],
        materialization_state="PARTIALLY_MATERIALIZED",
        overall_status="REJECTED_PROVENANCE",
        explicit_justification="Deterministically inventoryable rendered frames are insufficient without identity, license, label semantics or authoritative grouping.",
        source_identity_status="SOURCE_IDENTITY_UNVERIFIED",
        inventory_status="DETERMINISTIC_METADATA_INVENTORY",
        label_semantics_status="LABEL_SEMANTICS_UNVERIFIED",
        grouping_status="GROUP_PROVENANCE_UNVERIFIED",
        safe_reader_documentation_status="NOT_VERIFIABLE",
        official_source_or_limitation="Exact-name and filename-prefix searches did not establish a primary source.",
        evidence_category="LOCALLY_MEASURED",
    ),
    candidate(
        "local_sdt_zenodo_4124309",
        official_dataset_name="SDT Dataset | Synthetic Depth & Thermal Dataset for Person Detection and Pose Classification",
        stable_identifier="doi:10.5281/zenodo.4124309",
        official_distribution_location="https://zenodo.org/records/4124309",
        original_publication="https://doi.org/10.1109/ICIP40778.2020.9191284",
        license_terms="CONFLICT: Zenodo metadata CC-BY-4.0; record description says non-commercial research purposes only",
        research_use_permission="NON_COMMERCIAL_RESEARCH_EXPLICITLY_PERMITTED",
        model_training_permission="SYNTHETIC TRAIN/VALIDATION EXPLICITLY INTENDED FOR MODEL TRAINING; NON_COMMERCIAL LIMIT APPLIES",
        redistribution_restrictions="NEEDS_MANUAL_REVIEW_DUE_TO_OFFICIAL_TERM_CONFLICT",
        access_registration_requirements="OPEN_ACCESS; NO_REGISTRATION",
        genuine_thermal_status="MIXED: 8,000 real test thermal/depth pairs; 40,000 synthetic train/validation pairs",
        rgb_colorized_only=False,
        representation_classification="MULTIMODAL",
        representation_details={"image_t": "RADIOMETRIC_TEMPERATURE in Kelvin using FLIR 16/14-bit encoding", "image_d": "DEPTH in millimetres"},
        sensor_model={"real_thermal": "FLIR Lepton 3.5", "real_depth": "Orbbec Astra", "synthetic": "Blender rendering plus camera-specific noise"},
        wavelength="UNKNOWN_IN_REVIEWED_OFFICIAL_DOCUMENTATION",
        source_frame_shape=[480, 640, 1],
        source_orientation="ELEVATED_CAMERA; thermal native 160x120 bilinearly upscaled to 640x480",
        dtype="uint16 PNG",
        bit_depth="16-bit container; thermal values documented as FLIR 16/14-bit encoding",
        channels=1,
        file_format="PNG plus labels.txt",
        frame_rate="NOT_APPLICABLE_STATIC_IMAGE_PAIRS",
        timestamp_availability="ABSENT",
        subject_identifiers="ABSENT",
        session_identifiers="ABSENT",
        scene_identifiers="FOUR ROOM TYPES AS LABEL BALANCING FACTOR, BUT NO PER-ROW SCENE ID DOCUMENTED",
        sequence_identifiers="ABSENT",
        event_identifiers="ABSENT",
        camera_identifiers="MODALITY PAIR ONLY",
        fall_labels="SOURCE LABELS: lying, sitting, standing and empty room; SafeNest maps lying to HUMAN_FALL only as a derived post-fall lying-posture proxy",
        fall_event_boundary_quality="NOT_APPLICABLE; STATIC POSE IMAGES",
        normal_activity_coverage="STATIC SITTING/STANDING ONLY",
        hard_negative_coverage="LYING, SITTING, STANDING, EMPTY ROOM ACROSS FOUR ROOM TYPES",
        staged_vs_natural_fall_semantics="NO TEMPORAL FALL EVENTS; real test contains uniquely posed postures including lying",
        subject_count="NOT_DOCUMENTED",
        session_count="NOT_APPLICABLE",
        sequence_count="NOT_APPLICABLE",
        event_count="ZERO TEMPORALLY ANNOTATED FALL EVENTS",
        subject_wise_split_feasibility="NO",
        fallback_grouping_feasibility="ACCEPTED LIMITATION: preserve the official synthetic train / synthetic validation / real test split exactly; never perform a frame-random resplit",
        duplicate_near_duplicate_risk="UNKNOWN; synthetic variants and static real poses require a later diagnostic",
        event_level_evaluation_compatibility="NOT_VERIFIABLE; usable for sensor-level lying/post-fall posture evidence, not temporal fall-onset or end evaluation",
        approximate_download_storage_impact="19,223,751,874 bytes for local split files; 19.2 GB official record total including documentation",
        checksum_availability="OFFICIAL MD5 FOR ALL FILES; local test MD5 and SHA-256 verified",
        known_limitations=[
            "The source lying label is a SafeNest post-fall posture proxy, not proof that a fall event occurred.",
            "No subject/session/sequence/event identifiers or timestamps are supplied.",
            "Train/validation are synthetic and test is real; the official split must remain intact.",
            "Event-level onset/end performance and subject-wise generalization are not verifiable.",
            "Official license metadata and record text conflict; the stricter non-commercial research restriction governs this selection.",
            "Train parts and validation are local cloud placeholders and require owner-authorized hydration before T-A1 reads them.",
        ],
        materialization_state="MIXED: test materialized; train parts and validation are dataless placeholders",
        overall_status="SELECTED",
        explicit_justification="Selected with limitations for SafeNest sensor-level posture classification: lying is usable as post-fall posture evidence whose persistence and multisensor corroboration determine risk. Source identity, thermal/depth encoding, official split inventory, checksums and safe-reader semantics are documented; event-level fall claims remain prohibited.",
        source_identity_status="VERIFIED",
        license_status="VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION",
        inventory_status="DETERMINISTIC_INVENTORY_WITH_OFFICIAL_CHECKSUMS",
        label_semantics_status="USABLE_DERIVED_POST_FALL_POSTURE_PROXY",
        grouping_status="ACCEPTED_OFFICIAL_SPLIT_LIMITATION",
        safe_reader_documentation_status="DOCUMENTED",
        official_source_or_limitation="Official Zenodo record, API metadata and readme reviewed.",
        evidence_category="OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
        safenest_sensor_role="POST_FALL_LYING_POSTURE_EVIDENCE; persistence and multisensor fusion escalate suspicion; no single thermal frame confirms a fall event",
        safenest_label_mapping={
            "0": {"source_label": "lying", "target_label": "HUMAN_FALL", "mapping_type": "DERIVED_POST_FALL_POSTURE_PROXY"},
            "1": {"source_label": "sitting", "target_label": "HUMAN_NORMAL", "mapping_type": "DIRECT_POSTURE_EVIDENCE"},
            "2": {"source_label": "standing", "target_label": "HUMAN_NORMAL", "mapping_type": "DIRECT_POSTURE_EVIDENCE"},
            "3": {"source_label": "empty room", "target_label": "NOT_HUMAN", "mapping_type": "DIRECT_PRESENCE_EVIDENCE"},
        },
    ),
    candidate(
        "external_thermal_fall_66",
        official_dataset_name="Thermal Fall 66: A robust dataset for thermal imaging-based fall detection and eldercare",
        stable_identifier="PII:S0952197625018214",
        official_distribution_location="DATA_AVAILABLE_ON_REQUEST_PER_PUBLISHER_PAGE",
        original_publication="https://www.sciencedirect.com/science/article/pii/S0952197625018214",
        license_terms="LICENSE_UNVERIFIED",
        research_use_permission="REQUEST_REQUIRED",
        model_training_permission="NOT_VERIFIABLE_UNTIL_TERMS_RECEIVED",
        redistribution_restrictions="NOT_VERIFIABLE",
        access_registration_requirements="AUTHOR_REQUEST_REQUIRED",
        genuine_thermal_status="PUBLICATION_CLAIMS_THERMAL_IMAGING; FILE REPRESENTATION NOT INSPECTED",
        representation_classification="UNKNOWN",
        fall_labels="PUBLICATION CLAIMS FALL SCENARIOS",
        normal_activity_coverage="PUBLICATION CLAIMS ELDERCARE SCENARIOS; DETAILS NOT VERIFIED",
        hard_negative_coverage="NOT_VERIFIABLE",
        staged_vs_natural_fall_semantics="NOT_VERIFIABLE_FROM PUBLIC LANDING PAGE",
        subject_count=66,
        subject_wise_split_feasibility="POTENTIAL; FILE-LEVEL IDENTIFIERS NOT VERIFIED",
        fallback_grouping_feasibility="NOT_VERIFIABLE",
        event_level_evaluation_compatibility="POTENTIAL_AFTER_ACCESS",
        checksum_availability="NOT_VERIFIABLE",
        known_limitations=["Data is available only on request.", "Public license, checksums and file schema were not verified.", "Representation and grouping fields cannot be audited without access."],
        overall_status="ACCESS_BLOCKED",
        explicit_justification="Potentially relevant but not selectable without access, license, representation and grouping evidence.",
        source_identity_status="VERIFIED_PUBLICATION_ONLY",
        license_status="LICENSE_UNVERIFIED",
        inventory_status="ACCESS_BLOCKED",
        label_semantics_status="PUBLICATION_ONLY",
        grouping_status="GROUP_PROVENANCE_UNVERIFIED",
        safe_reader_documentation_status="ACCESS_BLOCKED",
        official_source_or_limitation="Official publisher landing page states data will be made available on request.",
        evidence_category="OFFICIAL_EXTERNAL_SOURCE_VERIFIED",
    ),
]


LOCAL_ASSETS = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "inventory_date": ACCESS_DATE,
    "root_policy": "repository-relative POSIX paths only",
    "assets": [
        {
            "asset_id": "family_a_fall_non_fall",
            "path": "datasets/thermal/raw_frames/Thermal_Dataset_Fall_Non_Fall",
            "observation_source": ["OWNER_CONFIRMED_LOCAL_STATE", "LOCALLY_MEASURED"],
            "existence": "PATH_EXISTS",
            "git_visibility": False,
            "git_ignore_state": "GIT_IGNORED_PAYLOAD",
            "git_ignore_rule": "/datasets/thermal/",
            "materialization_state": "PARTIALLY_MATERIALIZED",
            "logical_size_bytes": 224906370,
            "allocated_size_bytes": 132087808,
            "locally_readable_status": "3723_READABLE_OFFLINE; 3025_LOCAL_CLOUD_PLACEHOLDER",
            "inventory_summary": {
                "file_count": 6748,
                "directory_count_including_root": 1,
                "extension_counts": {"png": 6748},
                "locally_allocated_files": 3723,
                "cloud_placeholder_files": 3025,
                "locally_verified_shapes": {"230x226": 3723},
                "locally_verified_modes": {"RGB": 3723},
                "prefix_counts": {"ILS_ILP": 2080, "ILS_SP": 1502, "SSJ_ILP": 1537, "SSJ_SP": 1629},
                "prefix_meanings": {"ILS": "UNKNOWN", "SSJ": "UNKNOWN", "ILP": "UNKNOWN", "SP": "UNKNOWN"},
                "prefix_materialization": {
                    "ILS_ILP": {"cloud_placeholder": 931, "locally_allocated": 1149, "logical_bytes": 71377251},
                    "ILS_SP": {"cloud_placeholder": 671, "locally_allocated": 831, "logical_bytes": 52903215},
                    "SSJ_ILP": {"cloud_placeholder": 705, "locally_allocated": 832, "logical_bytes": 51880018},
                    "SSJ_SP": {"cloud_placeholder": 718, "locally_allocated": 911, "logical_bytes": 48745886},
                },
            },
            "representation_status": "THERMAL_COLORIZED_RENDERING",
            "source_identity_status": "SOURCE_IDENTITY_UNVERIFIED",
            "license_status": "LICENSE_UNVERIFIED",
            "label_status": "LABEL_SEMANTICS_UNVERIFIED",
            "grouping_status": "GROUP_PROVENANCE_UNVERIFIED",
            "checksum_status": {
                "scope": "LOCALLY_ALLOCATED_FILES_ONLY",
                "method": "sha256(sorted(relative_path\\0logical_size\\0file_sha256\\n))",
                "sha256": "7634e110681c92baf2cbb94f95011ea560033ca6d4746881b437482bba2e9a7f",
                "excluded_cloud_placeholders": 3025,
            },
            "warnings": ["PARTIAL_MATERIALIZATION", "RGB_NOT_RADIOMETRIC", "PREFIX_MEANINGS_UNKNOWN"],
        },
        {
            "asset_id": "family_b_sdt_split_archives",
            "path": "datasets/raw_archives/thermal_split_zips",
            "observation_source": ["OWNER_CONFIRMED_LOCAL_STATE", "LOCALLY_MEASURED", "OFFICIAL_EXTERNAL_SOURCE_VERIFIED"],
            "existence": "PATH_EXISTS",
            "git_visibility": False,
            "git_ignore_state": "GIT_IGNORED_PAYLOAD",
            "git_ignore_rule": "/datasets/raw_archives/",
            "materialization_state": "MIXED",
            "logical_size_bytes": 19223751874,
            "allocated_size_bytes": 1740349440,
            "locally_readable_status": "test.zip READABLE_OFFLINE; train parts and validation.zip LOCAL_CLOUD_PLACEHOLDER",
            "inventory_summary": {
                "entries": [
                    {"name": "test.zip", "logical_size_bytes": 1740348425, "allocated_size_bytes": 1740349440, "materialization_state": "LOCALLY_MATERIALIZED", "readable_offline": True, "official_md5": "d59a739f3b5ecf373c94046fb94cd94f", "local_md5": "d59a739f3b5ecf373c94046fb94cd94f", "local_sha256": "3a838bd70835e579ecfaa820a6c0b4cbc6ba7b76729417c73845f0c959281449"},
                    {"name": "train.zip.001", "logical_size_bytes": 4194304000, "allocated_size_bytes": 0, "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "readable_offline": False, "official_md5": "a7dfe81a1db58219da14db966d75cb2e", "local_checksum": "NOT_COMPUTED_TO_AVOID_HYDRATION"},
                    {"name": "train.zip.002", "logical_size_bytes": 4194304000, "allocated_size_bytes": 0, "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "readable_offline": False, "official_md5": "5e56bf4c17a2ce2f4b5cb59881dd161e", "local_checksum": "NOT_COMPUTED_TO_AVOID_HYDRATION"},
                    {"name": "train.zip.003", "logical_size_bytes": 4194304000, "allocated_size_bytes": 0, "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "readable_offline": False, "official_md5": "5f4c46025a46139db311382aa709a3a1", "local_checksum": "NOT_COMPUTED_TO_AVOID_HYDRATION"},
                    {"name": "train.zip.004", "logical_size_bytes": 1408015891, "allocated_size_bytes": 0, "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "readable_offline": False, "official_md5": "fbc26a3785540ff269410cdc43d53eae", "local_checksum": "NOT_COMPUTED_TO_AVOID_HYDRATION"},
                    {"name": "validation.zip", "logical_size_bytes": 3492475558, "allocated_size_bytes": 0, "materialization_state": "LOCAL_CLOUD_PLACEHOLDER", "readable_offline": False, "official_md5": "5464368b4798b50c59de3e06599b2677", "local_checksum": "NOT_COMPUTED_TO_AVOID_HYDRATION"},
                ],
                "test_archive": {"member_count": 16002, "file_count": 16001, "directory_count": 1, "image_t_files": 8000, "image_d_files": 8000, "label_files": 1, "label_rows": 8000, "source_class_counts": {"0": 2000, "1": 2000, "2": 2000, "3": 2000}, "sample_image_t": {"shape": [480, 640], "dtype": "<u2", "mode": "I;16", "min": 29653, "max": 30599}, "sample_image_d": {"shape": [480, 640], "dtype": "<u2", "mode": "I;16", "min": 0, "max": 7001}},
            },
            "representation_status": "MULTIMODAL: image_t RADIOMETRIC_TEMPERATURE; image_d DEPTH",
            "source_identity_status": "VERIFIED_ZENODO_4124309",
            "license_status": "VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION",
            "label_status": "USABLE_POSE_LABELS; LYING_IS_DERIVED_POST_FALL_POSTURE_PROXY",
            "grouping_status": "ACCEPTED_OFFICIAL_SPLIT_LIMITATION",
            "checksum_status": "MATERIALIZED_TEST_BYTE_IDENTITY_VERIFIED; PLACEHOLDERS_USE_OFFICIAL_MD5_ONLY",
            "warnings": ["LYING_PROXY_DOES_NOT_PROVE_FALL_EVENT", "PRESERVE_OFFICIAL_SPLITS", "TRAIN_VALIDATION_SYNTHETIC", "NONCOMMERCIAL_RESEARCH_RESTRICTION", "LARGE_DOWNLOAD_AUTHORIZATION_REQUIRED"],
        },
        {
            "asset_id": "family_c_processed_npz",
            "path": "datasets/thermal/processed_thermal_80x62.npz",
            "observation_source": ["OWNER_CONFIRMED_LOCAL_STATE", "LOCALLY_MEASURED", "REPOSITORY_CODE_VERIFIED"],
            "existence": "PATH_EXISTS",
            "git_visibility": False,
            "git_ignore_state": "GIT_IGNORED_PAYLOAD",
            "git_ignore_rule": "/datasets/thermal/",
            "materialization_state": "LOCALLY_MATERIALIZED",
            "logical_size_bytes": 330777971,
            "allocated_size_bytes": 330780672,
            "locally_readable_status": "READABLE_OFFLINE",
            "inventory_summary": {"array_keys": ["X", "y"], "X": {"shape": [54218, 62, 80], "dtype": "<f4", "min": 0.0, "max": 1.0, "finite": True}, "y": {"shape": [54218], "dtype": "<i4", "class_counts": {"0": 12003, "1": 26792, "2": 15423}}},
            "representation_status": "NORMALIZED_THERMAL_MIXED_WITH_RENDERED_RGB_GRAYSCALE",
            "source_identity_status": "PROCESSED_LINEAGE_PARTIALLY_RECONSTRUCTED",
            "license_status": "MIXED_AND_UNVERIFIED",
            "label_status": "MIXED_SOURCE_LABELS_AND_HEURISTIC_LABELS",
            "grouping_status": "PROVENANCE_LOST",
            "checksum_status": {"sha256": "3d6ad1eb2ed0438f0faaf83abed8b6e2c175074dfa031dcb4a5739c45984d06e"},
            "warnings": ["NOT_T_A_CANONICAL", "PROCESSED_LINEAGE_UNVERIFIED", "ONLY_X_Y_RETAINED"],
        },
        {
            "asset_id": "family_d_additional_image_tree",
            "path": "datasets/thermal/thermal image",
            "observation_source": ["OWNER_CONFIRMED_LOCAL_STATE", "LOCALLY_MEASURED"],
            "existence": "PATH_EXISTS",
            "git_visibility": False,
            "git_ignore_state": "GIT_IGNORED_PAYLOAD",
            "git_ignore_rule": "/datasets/thermal/",
            "materialization_state": "PARTIALLY_MATERIALIZED",
            "logical_size_bytes": 98992778,
            "allocated_size_bytes": 57053184,
            "locally_readable_status": "ALL_410_JSON_READABLE; 213_OF_410_IMAGES_READABLE; 197_IMAGES_LOCAL_CLOUD_PLACEHOLDER",
            "inventory_summary": {"file_count_including_ds_store": 821, "directory_count_including_root": 3, "image_count": 410, "json_count": 410, "cloud_placeholder_images": 197, "locally_allocated_images": 213, "extensions": {"jpeg": 384, "json": 410, "png": 26, "no_extension": 1}, "class_tree_counts": {"human": {"images": 211, "json": 212}, "not human": {"images": 199, "json": 198}}, "locally_verified_image_modes": {"RGB": 199, "RGBA": 14}},
            "representation_status": "THERMAL_COLORIZED_RENDERING_SCREENSHOTS_OR_EXPORTS",
            "source_identity_status": "SOURCE_IDENTITY_UNVERIFIED",
            "license_status": "LICENSE_UNVERIFIED",
            "label_status": "VERIFIED_HUMAN_PRESENCE_ONLY_NOT_FALL",
            "grouping_status": "GROUP_PROVENANCE_UNVERIFIED",
            "checksum_status": {"scope": "LOCALLY_ALLOCATED_FILES_ONLY", "method": "sha256(sorted(relative_path\\0logical_size\\0file_sha256\\n))", "sha256": "7c0a198994553c5dfde09aa7ad868650baee35960645c8403b4052c3fdaa5643", "excluded_cloud_placeholders": 197},
            "warnings": ["PRESENCE_LABEL_NOT_FALL_LABEL", "RGB_RENDERING_NOT_RADIOMETRIC", "PARTIAL_MATERIALIZATION"],
        },
    ],
}


SELECTED = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "decision_date": ACCESS_DATE,
    "selected_candidate_id": "local_sdt_zenodo_4124309",
    "selection_status": "PASS_WITH_LIMITATIONS",
    "overall_decision": "LOCAL_DATASET_SELECTED_WITH_LIMITATIONS",
    "t_a1_authorized": True,
    "t_a1_authorization_reason": "SDT satisfies the T-A0 source-basis gate for SafeNest posture evidence when lying is explicitly treated as a derived post-fall posture proxy, the stricter non-commercial research restriction is observed, and the official synthetic-train/synthetic-validation/real-test split is preserved. Temporal fall-event performance remains not verifiable.",
    "minimum_selection_rules": {"verified_source_identity": True, "acceptable_license_and_terms": True, "known_representation": True, "deterministic_inventory": True, "usable_label_semantics": True, "usable_grouping_provenance_or_accepted_limitation": True, "safe_reader_without_guessing": True},
    "canonical_source_name": "SDT Dataset | Synthetic Depth & Thermal Dataset for Person Detection and Pose Classification",
    "official_source": "https://zenodo.org/records/4124309",
    "stable_identifier": "doi:10.5281/zenodo.4124309",
    "license": "Stricter common denominator: non-commercial research use with citation/attribution; Zenodo metadata separately states CC-BY-4.0",
    "permitted_use": "SafeNest non-commercial research and model training; raw redistribution and commercial use require separate terms review",
    "representation": {"image_t": "RADIOMETRIC_TEMPERATURE; Kelvin in documented FLIR 16/14-bit encoding", "image_d": "DEPTH; millimetres"},
    "sensor_info": {"real_thermal": "FLIR Lepton 3.5", "real_depth": "Orbbec Astra", "synthetic": "Blender plus documented sensor-noise simulation"},
    "grouping_unit": "OFFICIAL_SOURCE_SPLIT; synthetic train, synthetic validation, real test; accepted limitation because subject/session/event IDs are absent",
    "label_semantics": "Original lying/sitting/standing/empty labels are preserved. SafeNest derives lying -> HUMAN_FALL as post-fall lying-posture evidence, not a fall-event assertion.",
    "source_checksum_information": "Official MD5 exists for every split file; local test.zip MD5 matches and local SHA-256 is recorded; placeholder bytes are not locally rehashed.",
    "t_a1_eligibility": "YES_WITH_LIMITATIONS_AND_OWNER_AUTHORIZED_HYDRATION_BEFORE_READING_PLACEHOLDERS",
    "limitations": ["No temporal fall onset/end labels.", "No subject/session/sequence/event IDs.", "Synthetic train/validation versus real test domain gap.", "Non-commercial research restriction governs use.", "No frame-random resplit; do not use the legacy mixed NPZ as canonical input."],
    "candidate_specific_decisions": {
        "external_ehomeseniors_2019": "NEEDS_MANUAL_REVIEW",
        "external_muvim_2022": "ACCESS_BLOCKED",
        "external_thermal_fall_66": "ACCESS_BLOCKED",
        "local_additional_human_not_human": "REJECTED_PROVENANCE",
        "local_family_a_fall_non_fall_png": "REJECTED_PROVENANCE",
        "local_sdt_zenodo_4124309": "SELECTED",
    },
    "blockers": [],
    "limitations_requiring_enforcement": ["LARGE_DOWNLOAD_AUTHORIZATION_REQUIRED before placeholder hydration", "PRESERVE_OFFICIAL_SOURCE_SPLITS", "GENERALIZATION_PERFORMANCE_NOT_VERIFIABLE", "EVENT_LEVEL_FALL_PERFORMANCE_NOT_VERIFIABLE", "NONCOMMERCIAL_RESEARCH_USE_ONLY_UNLESS_TERMS_ARE_CLEARED"],
    "prohibited_progression": ["Do not start T-A1 on this branch.", "Do not create a split from processed_thermal_80x62.npz.", "Do not frame-randomly resplit SDT.", "Do not describe the lying proxy as proof of a fall event."],
}


LINEAGE = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "artifact_path": "datasets/thermal/processed_thermal_80x62.npz",
    "measured_sha256": "3d6ad1eb2ed0438f0faaf83abed8b6e2c175074dfa031dcb4a5739c45984d06e",
    "size_bytes": 330777971,
    "array_keys": ["X", "y"],
    "arrays": {"X": {"shape": [54218, 62, 80], "dtype": "<f4"}, "y": {"shape": [54218], "dtype": "<i4"}},
    "known_source_contributors": [
        {"source": "SDT test.zip image_t", "row_range": [40000, 47999], "row_count": 8000, "evidence": "Selected test indices 0,1,2,100,999,3999,7999 exactly match NPZ rows 40000,40001,40002,40100,40999,43999,47999 after current resize/min-max code; class distribution also matches.", "evidence_category": "LOCALLY_MEASURED"},
        {"source": "Family A colorized PNG files", "row_range": [48020, 54217], "row_count": 6198, "evidence": "Current code appends Family A after the additional tree; 3,448 locally allocated files exactly match rows in this range. Total row count is inferred from residual ordering.", "evidence_category": "INFERRED_WITH_LOCAL_SPOT_MATCHES"},
        {"source": "Additional human/not-human image tree", "row_range": [48000, 48019], "row_count": 20, "evidence": "Current code appends this tree before Family A; six locally allocated images exactly match rows 48002-48012 and the first Family A match begins at 48020.", "evidence_category": "INFERRED_WITH_LOCAL_SPOT_MATCHES"},
    ],
    "inferred_source_contributors": [
        {"source": "SDT train split image_t", "row_range": [0, 31999], "row_count": 32000, "reason": "Current code order, official split size, and exact balanced class counts 8000/16000/8000; train payload is a cloud placeholder so byte-level confirmation was not attempted."},
        {"source": "SDT validation split image_t", "row_range": [32000, 39999], "row_count": 8000, "reason": "Current code order, official split size, and exact balanced class counts 2000/4000/2000; validation payload is a cloud placeholder."},
    ],
    "unknown_source_contributors": ["Exact original file ID for most rows is absent.", "Why only 6,198 of 6,748 current Family A files appear in the artifact is unknown.", "Why only 20 of the current 410 additional images appear is unknown.", "Generation timestamp, source commit and artifact move from legacy thermal/ to datasets/thermal are not recorded."],
    "legacy_path_assumptions": {"preparation_input_sdt_primary": "../datasets/raw_archives/thermal_split_zips relative to project root (resolves outside canonical root and is erroneous for the active layout)", "preparation_input_sdt_fallback": "thermal_new_dataset", "small_image_input": "thermal/thermal image", "family_a_input": "thermal/archive/Thermal_Dataset_Fall_Non_Fall", "output": "thermal/processed_thermal_80x62.npz", "active_artifact": "datasets/thermal/processed_thermal_80x62.npz"},
    "preprocessing_evidence": {"sdt": ["read only image_t", "bilinear resize to 80x62", "per-frame min-max to [0,1]", "source pose-to-SafeNest class mapping"], "additional_tree": ["convert RGB/RGBA rendering to grayscale", "bilinear resize to 80x62", "divide by 255", "presence folder or geometry heuristic labels"], "family_a": ["convert RGB rendering to grayscale", "bilinear resize to 80x62", "divide by 255", "geometry heuristic labels"], "exception_behavior": "Multiple broad except Exception: pass blocks silently skip failures."},
    "split_history_evidence": {"source_split_merge": "CONFIRMED_IN_CODE: thermal_prep.py appends SDT train, validation and test into one X/y array.", "later_random_split": "CONFIRMED_IN_CODE: thermal_train.py applies a seeded frame-level permutation and 80:20 split to the combined array.", "execution_against_current_model": "NOT_VERIFIABLE: no immutable training log or model-to-NPZ lineage manifest proves the script execution that produced the current TFLite artifact.", "risk": "If executed as written, original split isolation and subject/event grouping are destroyed, creating source/split contamination risk."},
    "provenance_retention": {"retained": ["normalized frame tensor", "derived class integer"], "lost": ["source dataset", "source split", "source file", "frame index", "subject", "session", "scene", "sequence", "event", "camera", "timestamp", "original label", "mapping type", "quality/exclusion record"]},
    "confidence_status": "PROCESSED_LINEAGE_PARTIALLY_RECONSTRUCTED",
    "canonical_status": "LEGACY_OR_GENERATED_EXISTING_ARTIFACT; NOT_T_A_CANONICAL",
}


LIMITATIONS = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "overall_outcome": "PASS_WITH_LIMITATIONS",
    "limitations": [
        {"id": "T-A0-L001", "status": "OPEN", "issue": "Family A identity, license, labels and prefix meanings are unverified."},
        {"id": "T-A0-L002", "status": "ACCEPTED_LIMITATION", "issue": "SDT lying is mapped to HUMAN_FALL only as derived post-fall lying-posture evidence; it never proves a temporal fall event by itself."},
        {"id": "T-A0-L003", "status": "ACCEPTED_RESTRICTIVE_COMMON_DENOMINATOR", "issue": "SDT official CC-BY-4.0 metadata conflicts with non-commercial-research-only record text; this selection applies the stricter non-commercial research, citation and attribution conditions."},
        {"id": "T-A0-L004", "status": "ACCEPTED_GROUPING_LIMITATION", "issue": "SDT has no subject/session/event IDs. T-A1 must preserve the official synthetic train / synthetic validation / real test split exactly; subject-wise and event-level generalization remain not verifiable."},
        {"id": "T-A0-L005", "status": "OPEN", "issue": "Legacy NPZ retains only X/y and mixes official splits and heuristic labels."},
        {"id": "T-A0-L006", "status": "LARGE_DOWNLOAD_AUTHORIZATION_REQUIRED", "issue": "SDT train and validation payloads are dataless multi-GB placeholders."},
        {"id": "T-A0-L007", "status": "ACCESS_BLOCKED", "issue": "MUVIM and Thermal Fall 66 require external access and terms review."},
        {"id": "T-A0-L008", "status": "OPEN", "issue": "eHomeSeniors lacks documented normal/ADL sequences and explicit repeated-fall boundaries."},
        {"id": "T-A0-L009", "status": "DEFERRED_T_C", "issue": "Thermal-44 units, dtype, endianness, conversion, invalid pixels, packet size and real driver remain unverified."},
    ],
    "documentation_discrepancy": {"roadmap_claim": "Thermal real evaluation dataset absent", "corrected_interpretation": "NO_APPROVED_CANONICAL_REAL_THERMAL_EVALUATION_DATASET", "local_payload_exists": True, "roadmap_modified": False},
}


SOURCE_EVIDENCE = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "access_date": ACCESS_DATE,
    "sources": [
        {"source_id": "ehomeseniors_paper", "url": "https://doi.org/10.3390/s19204565", "category": "OFFICIAL_PUBLICATION", "verified_claims": ["six volunteers", "two thermal sensors", "CSV/MAT schema", "staged-fall protocol", "subject-bearing file names", "publisher supplement location"]},
        {"source_id": "muvim_paper", "url": "https://doi.org/10.1109/ACCESS.2022.3211939", "category": "OFFICIAL_PUBLICATION", "verified_claims": ["thermal/IR/depth/RGB modalities", "30 younger fall/ADL subjects", "10 older ADL subjects", "three FLIR ONE Gen 3 cameras", "author-request access"]},
        {"source_id": "sdt_api", "url": "https://zenodo.org/api/records/4124309", "category": "OFFICIAL_DATASET_API", "verified_claims": ["DOI", "version", "file sizes", "official MD5 values", "CC-BY-4.0 metadata", "non-commercial research text"]},
        {"source_id": "sdt_readme", "url": "https://zenodo.org/records/4124309/files/readme.md?download=1", "category": "OFFICIAL_DATASET_DOCUMENTATION", "verified_claims": ["image_t thermal", "image_d depth", "Kelvin encoding", "depth millimetres", "sensor models", "pose labels", "split sizes", "synthetic versus real splits"]},
        {"source_id": "thermal_fall_66_publisher", "url": "https://www.sciencedirect.com/science/article/pii/S0952197625018214", "category": "OFFICIAL_PUBLICATION_LANDING_PAGE", "verified_claims": ["66-subject thermal fall dataset claim", "data available on request"]},
    ],
    "license_decisions": [
        {"candidate_id": "local_sdt_zenodo_4124309", "status": "VERIFIED_ACCEPTABLE_WITH_NONCOMMERCIAL_RESEARCH_RESTRICTION", "reason": "The official record explicitly permits non-commercial research and documents synthetic training use. The stricter statement governs; attribution/citation are required, and commercial use or raw redistribution needs separate review."},
        {"candidate_id": "external_ehomeseniors_2019", "status": "NEEDS_MANUAL_REVIEW", "reason": "CC BY 4.0 covers the article/publisher supplement context, but a separate dataset license statement was not found."},
        {"candidate_id": "external_muvim_2022", "status": "LICENSE_UNVERIFIED", "reason": "Terms require access request."},
        {"candidate_id": "external_thermal_fall_66", "status": "LICENSE_UNVERIFIED", "reason": "Terms require access request."},
    ],
}


MODEL_AUDIT = {
    "schema_version": "1.0",
    "phase": "T-A0",
    "artifact_path": "models/thermal/thermal_fall_int8_v0.1.0.tflite",
    "measured_size_bytes": 318184,
    "measured_sha256": "5b56da8d127ccef85f30b6459cc0cfe2d86490e41f3caa5bd2a7b70bbc46ae84",
    "manifest_match": True,
    "input_tensor": {"shape": [1, 62, 80, 1], "dtype": "int8", "scale": 0.003921568859368563, "zero_point": -128},
    "output_tensor": {"shape": [1, 3], "dtype": "int8", "scale": 0.00390625, "zero_point": -128},
    "class_map": {"0": "NOT_HUMAN", "1": "HUMAN_NORMAL", "2": "HUMAN_FALL"},
    "preprocessing": "per-frame min-max normalization when values fall outside [0,1]",
    "absolute_celsius_information": "DISCARDED_BY_NORMALIZATION",
    "validation_claim": "ARTIFACT_AND_SOFTWARE_CONTRACT_ONLY; NO_NEW_MODEL_PERFORMANCE_CLAIM",
    "thermal44_hardware_status": "NOT_VERIFIABLE_DEFERRED_TO_T_C",
}


VALIDATION_RESULT = {
    "candidate_count": 6,
    "error_count": 0,
    "errors": [],
    "evidence_validation": "PASS",
    "local_asset_count": 4,
    "overall_outcome": "PASS_WITH_LIMITATIONS",
    "phase": "T-A0",
    "schema_version": "1.0",
    "selected_candidate_id": "local_sdt_zenodo_4124309",
    "t_a1_authorized": True,
    "warning_count": 0,
    "warnings": [],
}


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(name: str, data: Any) -> Path:
    path = OUT / name
    path.write_text(canonical_json(data), encoding="utf-8")
    return path


def render_report() -> str:
    return f"""# SafeNest Thermal T-A0 Dataset Selection and Source Identity

- Phase: `T-A0`
- Audit date: `{ACCESS_DATE}`
- Overall outcome: `PASS_WITH_LIMITATIONS`
- Selection status: `LOCAL_DATASET_SELECTED_WITH_LIMITATIONS`
- T-A1 authorized: `YES`

## Decision

The local SDT source (`doi:10.5281/zenodo.4124309`) is selected with explicit limitations as the T-A1 source basis. The local payload exists and is intentionally Git-ignored; absence from Git is not absence from the owner workspace.

SDT source label 0 remains **lying**. SafeNest derives it as `HUMAN_FALL` only in the narrower sense of **post-fall lying-posture evidence**: a single frame does not establish that a fall event occurred. Persistence and corroboration from other sensors are responsible for escalating suspicion. This matches the intended sensor-fusion architecture while preserving the original source semantics.

SDT has no subject/session/sequence/event identifiers, so T-A1 must preserve its official synthetic-train, synthetic-validation and real-test split exactly. Subject-wise and event-level generalization are `NOT_VERIFIABLE`, and frame-random resplitting is prohibited. Family A and the additional human/not-human tree remain unselected because their source provenance is insufficient. The processed NPZ remains legacy mixed-source evidence and is not canonical.

## Candidate comparison

| Candidate | Representation | Label/group evidence | Access/license | T-A0 status |
|---|---|---|---|---|
| Local Family A | RGB thermal colorized rendering | Unknown labels and grouping | Identity/license unknown | `REJECTED_PROVENANCE` |
| Local SDT | 16-bit thermal Kelvin encoding + depth; synthetic train/validation, real test | Lying as derived post-fall posture proxy; official split is the accepted grouping limitation | Non-commercial research restriction, citation/attribution; official metadata conflict retained | `SELECTED` |
| Local human/not-human tree | RGB/RGBA thermal screenshots/exports | Presence polygons only | Identity/license unknown | `REJECTED_PROVENANCE` |
| eHomeSeniors | Numeric thermal temperature and raw fields | Six subjects and staged fall types; no documented normal sequences or explicit repeated-event boundaries | Open supplement; dataset-specific terms need review | `NEEDS_MANUAL_REVIEW` |
| MUVIM | Encoded thermal video plus other modalities | Strong publication-level subject/ADL/fall structure | Author request; terms unverified | `ACCESS_BLOCKED` |
| Thermal Fall 66 | Thermal representation not inspectable | Publication claims 66 participants | Author request; terms unverified | `ACCESS_BLOCKED` |

## Local inventory

- Family A: 6,748 PNG, 224,906,370 logical bytes; 3,723 readable RGB 230×226 files and 3,025 dataless placeholders.
- SDT: `test.zip` is materialized and byte-identical to official MD5; it contains 8,000 `image_t`, 8,000 `image_d`, and 8,000 five-field labels. Four train parts and validation are dataless placeholders. No large hydration was attempted.
- Processed NPZ: 330,777,971 bytes, SHA-256 `3d6ad1eb2ed0438f0faaf83abed8b6e2c175074dfa031dcb4a5739c45984d06e`; only `X` `(54218,62,80)` float32 and `y` `(54218,)` int32 survive.
- Additional tree: 410 images and 410 JSON annotations; all JSON and 213 images are readable, while 197 images are dataless placeholders.

## Processed NPZ lineage

Selected SDT test samples exactly match NPZ rows 40,000–47,999 under the current preparation transform. Code order, segment counts and local spot matches support a partial reconstruction of 32,000 SDT train + 8,000 SDT validation + 8,000 SDT test + 20 additional-tree images + 6,198 Family A images. Exact per-row source IDs, generation commit, skip reasons and original grouping are absent, so the artifact remains `PROCESSED_LINEAGE_PARTIALLY_RECONSTRUCTED` and `NOT_T_A_CANONICAL`.

`thermal_prep.py` merges original train/validation/test sources and silently swallows broad exceptions. `thermal_train.py` then defines a seeded frame-level 80:20 permutation. This is confirmed code risk; execution against the current TFLite artifact is not independently proven by an immutable training record.

## Contract boundaries preserved

Per-frame min-max normalization discards absolute Celsius context. Thermal-44 physical unit, dtype, endianness, raw-count conversion, invalid pixels, 9,920-versus-10,080 bytes, real driver and hardware/Pi evidence remain `NOT_VERIFIABLE` and deferred to `T-C`. No T-A1 split, tensor regeneration, training or model-performance claim was created.

## T-A1 gate

`T-A1 authorized: YES`, with these mandatory conditions: use SDT under the stricter non-commercial research and attribution terms; obtain owner authorization before hydrating multi-GB placeholders; read the original archives rather than the mixed legacy NPZ; preserve the official train/validation/test split; retain the source labels and derived-proxy mapping in row provenance; and do not claim temporal fall-event, subject-generalization, Thermal-44 hardware or model-performance validation from this T-A0 decision.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        write_json("candidate_registry.json", {"schema_version": "1.0", "phase": "T-A0", "access_date": ACCESS_DATE, "candidates": sorted(CANDIDATES, key=lambda x: x["candidate_id"]), "excluded_derived_artifacts": [{"path": "datasets/thermal/processed_thermal_80x62.npz", "reason": "Derived legacy artifact; audited in processed_lineage.json rather than treated as a source."}]}),
        write_json("limitations.json", LIMITATIONS),
        write_json("local_asset_registry.json", LOCAL_ASSETS),
        write_json("model_artifact_audit.json", MODEL_AUDIT),
        write_json("processed_lineage.json", LINEAGE),
        write_json("selected_source_identity.json", SELECTED),
        write_json("source_license_evidence.json", SOURCE_EVIDENCE),
        write_json("validation_result.json", VALIDATION_RESULT),
    ]
    REPORT.write_text(render_report(), encoding="utf-8")
    files.append(REPORT)
    lines = []
    for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
