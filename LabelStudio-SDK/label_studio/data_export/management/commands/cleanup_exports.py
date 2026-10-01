"""Delete old export snapshots and the files they keep on disk.

Export snapshots (and the format conversions made from them) live under
``settings.DELAYED_EXPORT_DIR`` and are never removed automatically, so a project
that is exported regularly grows the media volume without bound. Run this on
whatever schedule suits the deployment; it is not wired to a timer.

Exports that are still running are always skipped, whatever their age.

Examples:
    # Show what would be removed, without changing anything
    python manage.py cleanup_exports --days 30 --dry-run

    # Remove export snapshots older than 30 days
    python manage.py cleanup_exports --days 30

    # Only for one project, and keep a week's worth
    python manage.py cleanup_exports --days 7 --project 266281
"""

import logging
from datetime import timedelta

from data_export.models import Export
from django.core.management.base import BaseCommand
from django.template.defaultfilters import filesizeformat
from django.utils import timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Delete export snapshots older than --days and the files they stored'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Delete exports created more than this many days ago (default: 30)',
        )
        parser.add_argument(
            '--project',
            type=int,
            default=None,
            help='Only clean up exports belonging to this project ID',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would be deleted without deleting anything',
        )

    def handle(self, *args, **options):
        days = options['days']
        project_id = options['project']
        dry_run = options['dry_run']

        if days < 0:
            self.stderr.write(self.style.ERROR('--days must not be negative'))
            return

        cutoff = timezone.now() - timedelta(days=days)

        queryset = Export.objects.filter(created_at__lt=cutoff).exclude(status=Export.Status.IN_PROGRESS)
        if project_id is not None:
            queryset = queryset.filter(project_id=project_id)

        queryset = queryset.prefetch_related('converted_formats')

        total = queryset.count()
        if total == 0:
            self.stdout.write('Nothing to clean up.')
            return

        freed = 0
        deleted = 0
        for export in queryset.iterator(chunk_size=200):
            freed += self._file_size(export.file)
            for converted in export.converted_formats.all():
                freed += self._file_size(converted.file)

            if dry_run:
                continue

            # Rows cascade on delete but files do not, so remove them first.
            self._delete_file(export.file)
            for converted in export.converted_formats.all():
                self._delete_file(converted.file)
            export.delete()
            deleted += 1

        freed_display = filesizeformat(freed)
        if dry_run:
            self.stdout.write(
                f'Would delete {total} export(s) created before {cutoff:%Y-%m-%d}, freeing {freed_display}.'
            )
        else:
            self.stdout.write(self.style.SUCCESS(f'Deleted {deleted} export(s), freeing {freed_display}.'))

    @staticmethod
    def _file_size(field_file):
        if not field_file:
            return 0
        try:
            return field_file.size
        except (OSError, ValueError):
            # Storage may already have lost the file; nothing to account for.
            return 0

    @staticmethod
    def _delete_file(field_file):
        if not field_file:
            return
        try:
            field_file.delete(save=False)
        except OSError:
            logger.warning('Could not delete export file %s', field_file.name, exc_info=True)
