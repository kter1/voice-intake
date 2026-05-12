"""Audit chain fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-23
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audit_events",
        sa.Column("chain_hash", sa.String(), nullable=False, server_default=""),
    )
    op.add_column(
        "audit_events",
        sa.Column("chain_index", sa.Integer(), nullable=True),
    )
    op.create_unique_constraint(
        "uq_audit_events_session_chain",
        "audit_events",
        ["session_id", "chain_index"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_audit_events_session_chain", "audit_events", type_="unique")
    op.drop_column("audit_events", "chain_index")
    op.drop_column("audit_events", "chain_hash")
