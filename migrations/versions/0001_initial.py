"""create jobs and leaderboard_entries tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01 00:00:00

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_type",
            sa.Enum("training", "evaluation", "dataset_download", name="jobtype"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "succeeded", "failed", "cancelled", name="jobstatus"),
            nullable=False,
        ),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=True),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "leaderboard_entries",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("dataset_name", sa.String(length=128), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("action_mse", sa.Float(), nullable=False),
        sa.Column("latency_p95_ms", sa.Float(), nullable=False),
        sa.Column("success_rate", sa.Float(), nullable=True),
        sa.Column("num_samples", sa.Integer(), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_leaderboard_entries_model_name", "leaderboard_entries", ["model_name"])
    op.create_index("ix_leaderboard_entries_dataset_name", "leaderboard_entries", ["dataset_name"])


def downgrade() -> None:
    op.drop_index("ix_leaderboard_entries_dataset_name", table_name="leaderboard_entries")
    op.drop_index("ix_leaderboard_entries_model_name", table_name="leaderboard_entries")
    op.drop_table("leaderboard_entries")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")
    sa.Enum(name="jobtype").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="jobstatus").drop(op.get_bind(), checkfirst=True)
