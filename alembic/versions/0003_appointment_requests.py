"""appointment_requests table for demo scheduling flow

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "appointment_requests",
        sa.Column("appointment_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("reason_for_visit", sa.Text(), nullable=True),
        sa.Column("patient_name", sa.String(), nullable=True),
        sa.Column("date_of_birth", sa.String(), nullable=True),
        sa.Column("insurance_name", sa.String(), nullable=True),
        sa.Column(
            "insurance_network_status",
            sa.String(),
            nullable=False,
            server_default="unknown",
        ),
        sa.Column(
            "appointment_status",
            sa.String(),
            nullable=False,
            server_default="collecting",
        ),
        sa.Column("scheduled_slot", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("appointment_id"),
        sa.UniqueConstraint("session_id", name="uq_appointment_requests_session"),
    )
    op.create_index(
        "ix_appointment_requests_session_id",
        "appointment_requests",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_appointment_requests_session_id", table_name="appointment_requests")
    op.drop_table("appointment_requests")
