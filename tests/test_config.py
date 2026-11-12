from demand_copilot import config


def test_project_folders_exist():
    for folder in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR, config.MODELS_DIR):
        assert folder.is_dir(), f"Missing folder {folder}"
