"""
Unit tests for `har_system.utils.cli`.

These tests validate CLI argument defaults and parsing in isolation, without running any
GStreamer pipelines or accessing external resources.
"""

from __future__ import annotations

import argparse

import pytest

from har_system.utils.cli import (
    add_realtime_arguments,
    add_faces_arguments,
    add_chokepoint_arguments,
    add_train_faces_arguments,
    build_realtime_parser,
)


@pytest.mark.unit
def test_realtime_parser_defaults():
    parser = build_realtime_parser()
    args = parser.parse_args([])

    assert args.input == "rpi"
    assert args.show_fps is False
    assert args.verbose is False
    assert args.save_data is False
    assert args.output_dir == "./results/camera"
    assert args.print_interval == 30
    assert args.enable_face_recognition is False
    # default is None so the app can read config/default.yaml
    assert args.database_dir is None
    assert args.no_display is False


@pytest.mark.unit
def test_realtime_parser_parses_flags():
    parser = build_realtime_parser()
    args = parser.parse_args(
        [
            "--input",
            "usb",
            "--show-fps",
            "--verbose",
            "--save-data",
            "--output-dir",
            "./out",
            "--print-interval",
            "60",
            "--enable-face-recognition",
            "--database-dir",
            "./db",
            "--no-display",
        ]
    )
    assert args.input == "usb"
    assert args.show_fps is True
    assert args.verbose is True
    assert args.save_data is True
    assert args.output_dir == "./out"
    assert args.print_interval == 60
    assert args.enable_face_recognition is True
    assert args.database_dir == "./db"
    assert args.no_display is True


@pytest.mark.unit
def test_add_faces_arguments_supports_stats_and_list():
    parser = argparse.ArgumentParser()
    add_faces_arguments(parser)

    args = parser.parse_args(["--stats", "--list", "--database-dir", "./database"])
    assert args.stats is True
    assert args.list is True
    assert args.database_dir == "./database"


@pytest.mark.unit
def test_add_train_faces_arguments_defaults():
    parser = argparse.ArgumentParser()
    add_train_faces_arguments(parser)
    args = parser.parse_args([])
    assert args.train_dir == "./train_faces"
    assert args.database_dir == "./database"
    assert args.confidence_threshold == 0.70


@pytest.mark.unit
def test_add_chokepoint_arguments_defaults():
    parser = argparse.ArgumentParser()
    add_chokepoint_arguments(parser)
    args = parser.parse_args([])
    assert args.dataset_path == "./test_dataset"
    assert args.results_dir == "./results"
    assert args.enable_face_recognition is False
    assert args.database_dir == "./database"
    assert args.no_display is False

