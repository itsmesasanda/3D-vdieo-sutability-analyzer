from video_suitability.config import AnalysisConfig
from video_suitability.geometry import analyze_geometry
from video_suitability.synthetic import make_orbit_frames


def test_orbit_has_more_motion_than_stationary_sequence():
    frames = make_orbit_frames(width=360, height=270, frame_count=12)
    stationary = [frames[6].copy() for _ in frames]
    indices = list(range(len(frames)))
    timestamps = [i / 12.0 for i in indices]
    config = AnalysisConfig(max_samples=12, max_analysis_side=360)

    orbit_result = analyze_geometry(frames, indices, timestamps, config)["summary"]
    still_result = analyze_geometry(stationary, indices, timestamps, config)["summary"]

    assert orbit_result["valid_geometry_pair_fraction"] > 0.5
    assert orbit_result["median_displacement_normalized"] > 0.003
    assert still_result["near_stationary_pair_fraction"] >= 0.8
    assert orbit_result["median_displacement_normalized"] > still_result[
        "median_displacement_normalized"
    ]
    assert orbit_result["median_flow_track_retention"] > 0
    assert orbit_result["median_flow_speed_normalized_per_second"] > 0
    assert still_result["median_flow_speed_normalized_per_second"] < 0.01

