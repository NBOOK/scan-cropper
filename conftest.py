import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest


@pytest.fixture(autouse=True)
def no_blocking_message_boxes(monkeypatch):
    """An unexpected modal dialog would hang offscreen tests; fail instead."""
    from PySide6.QtWidgets import QMessageBox
    for name in ('warning','critical','information'):
        def unexpected(*args,_name=name,**kwargs):
            pytest.fail(f'unexpected QMessageBox.{_name}: {args[1:3]}')
        monkeypatch.setattr(QMessageBox,name,unexpected)


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path,monkeypatch):
    import scan_cropper_gui
    monkeypatch.setattr(scan_cropper_gui,'application_settings_path',lambda:tmp_path/'user-settings.json')
