"""Tests for the opt-in background export (``background: true`` on POST /exports/).

Exports normally run inside the creating request, so a project big enough to
outlive a proxy timeout cannot be exported from the UI at all. The flag moves the
work onto the in-process pool instead, which ``run_file_exporting`` falls back to
because the OSS build never enables Redis.
"""

import threading
import time
from datetime import timedelta
from unittest.mock import patch

from data_export.models import Export
from django.utils import timezone
from projects.tests.factories import ProjectFactory
from rest_framework.test import APITransactionTestCase


def _record_thread(target):
    """Return a side effect that stores the calling thread's ident in ``target``."""

    def side_effect(*args, **kwargs):
        target['thread'] = threading.get_ident()

    return side_effect


class TestBackgroundExport(APITransactionTestCase):
    """TransactionTestCase on purpose: the worker runs on its own thread with its
    own database connection, so the Export row has to be really committed for it
    to be found. TestCase's wrapping transaction would hide it."""

    def setUp(self):
        # Per test rather than setUpTestData: it is not run for transaction
        # test cases, and every test here needs committed rows anyway.
        self.project = ProjectFactory()
        self.client.force_authenticate(user=self.project.created_by)
        self.url = f'/api/projects/{self.project.id}/exports/'

    def _post(self, payload=None):
        return self.client.post(self.url, payload or {}, format='json')

    def _wait_for(self, export_id, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            export = Export.objects.get(id=export_id)
            if export.status in (Export.Status.COMPLETED, Export.Status.FAILED):
                return export
            time.sleep(0.05)
        self.fail(f'Export {export_id} never reached a final status')

    def test_without_the_flag_the_export_runs_on_the_request_thread(self):
        seen = {}

        with patch.object(Export, 'export_to_file', side_effect=_record_thread(seen)):
            response = self._post()

        assert response.status_code == 201
        assert 'thread' in seen, 'export_to_file was never called'
        assert seen['thread'] == threading.get_ident()

    def test_background_export_answers_while_the_worker_is_still_running(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        seen = {}

        def blocking_export(*args, **kwargs):
            _record_thread(seen)(*args, **kwargs)
            entered.set()
            release.wait(timeout=10)
            finished.set()

        with patch.object(Export, 'export_to_file', side_effect=blocking_export):
            response = self._post({'background': True})

            # The response is already out although the worker is parked inside
            # export_to_file. That is the entire point of the flag: the request no
            # longer lives as long as the export does.
            assert response.status_code == 201
            assert entered.wait(timeout=5), 'the export never reached the worker'
            assert response.json()['status'] == Export.Status.IN_PROGRESS
            assert seen['thread'] != threading.get_ident()

            release.set()
            assert finished.wait(timeout=5), 'the worker never finished'

    def test_background_export_completes_and_downloads(self):
        response = self._post({'background': True})

        assert response.status_code == 201
        export_id = response.json()['id']

        export = self._wait_for(export_id)
        assert export.status == Export.Status.COMPLETED, export.counters
        assert export.finished_at is not None

        download = self.client.get(f'{self.url}{export_id}/download?exportType=JSON')
        assert download.status_code == 200

    def test_export_left_in_progress_by_a_dead_process_is_failed(self):
        """The pool dies with the web process, so a restart would otherwise leave
        the UI polling a row that can never finish."""
        stale = Export.objects.create(project=self.project, status=Export.Status.IN_PROGRESS)
        # created_at is auto_now_add, so it has to be moved back explicitly to
        # look like it predates this process.
        Export.objects.filter(id=stale.id).update(created_at=timezone.now() - timedelta(days=1))

        # The sweep runs once per process; let it run again for this test.
        with patch('data_export.mixins._stale_exports_swept', False):
            response = self.client.get(self.url)

        assert response.status_code == 200
        stale.refresh_from_db()
        assert stale.status == Export.Status.FAILED
        assert stale.finished_at is not None
