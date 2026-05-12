"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-21
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String, primary_key=True),
        sa.Column("telephony_call_id", sa.String, nullable=False),
        sa.Column("operator_id", sa.String, nullable=False),
        sa.Column("current_state", sa.String, nullable=False),
        sa.Column("mode", sa.String, nullable=False),
        sa.Column("policy_profile_id", sa.String, nullable=False),
        sa.Column("template_bundle_version", sa.String, nullable=False),
        sa.Column("language_status", sa.String, nullable=False),
        sa.Column("stir_shaken_attestation", sa.String, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disposition", sa.String, nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("reject_counts_json", sa.Text, nullable=False, server_default="{}"),
    )
    op.create_table(
        "voice_turns",
        sa.Column("turn_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("speaker", sa.String, nullable=False),
        sa.Column("audio_ref", sa.String, nullable=True),
        sa.Column("retention_policy_id", sa.String, nullable=False),
        sa.Column("transcript", sa.Text, nullable=False),
        sa.Column("partial_or_final", sa.String, nullable=False),
        sa.Column("asr_confidence", sa.Float, nullable=False),
        sa.Column("barge_in", sa.Integer, nullable=False),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_voice_turns_session_id", "voice_turns", ["session_id"])
    op.create_table(
        "model_proposals",
        sa.Column("proposal_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("source_turn_id", sa.String, nullable=False),
        sa.Column("template_id", sa.String, nullable=False),
        sa.Column("variables_json", sa.Text, nullable=False, server_default="{}"),
        sa.Column("requested_transition", sa.String, nullable=False),
        sa.Column("model_version", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_model_proposals_session_id", "model_proposals", ["session_id"])
    op.create_table(
        "validator_results",
        sa.Column("validator_result_id", sa.String, primary_key=True),
        sa.Column("proposal_id", sa.String, sa.ForeignKey("model_proposals.proposal_id"), nullable=False),
        sa.Column("accepted", sa.Integer, nullable=False),
        sa.Column("reject_reason", sa.String, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_validator_results_proposal_id", "validator_results", ["proposal_id"])
    op.create_table(
        "audit_events",
        sa.Column("event_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("actor", sa.String, nullable=False),
        sa.Column("event_type", sa.String, nullable=False),
        sa.Column("proposal_id", sa.String, nullable=True),
        sa.Column("validator_result_id", sa.String, nullable=True),
        sa.Column("affected_fields_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("identity_state", sa.String, nullable=False),
        sa.Column("policy_decision_id", sa.String, nullable=True),
        sa.Column("template_id", sa.String, nullable=True),
        sa.Column("model_version", sa.String, nullable=True),
        sa.Column("service_path", sa.String, nullable=False),
        sa.Column("before_after_hash", sa.String, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details_json", sa.Text, nullable=False, server_default="{}"),
    )
    op.create_index("ix_audit_events_session_id", "audit_events", ["session_id"])
    op.create_table(
        "consent_artifacts",
        sa.Column("artifact_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("consent_type", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("jurisdiction_policy_id", sa.String, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("capture_mode", sa.String, nullable=False),
        sa.Column("storage_effect", sa.String, nullable=False),
    )
    op.create_index("ix_consent_artifacts_session_id", "consent_artifacts", ["session_id"])
    op.create_table(
        "supervisor_interventions",
        sa.Column("intervention_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("intervention_type", sa.String, nullable=False),
        sa.Column("reason_code", sa.String, nullable=False),
        sa.Column("notes", sa.Text, nullable=False),
        sa.Column("alert_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ack_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("takeover_ts", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_supervisor_interventions_session_id", "supervisor_interventions", ["session_id"])
    op.create_table(
        "policy_decisions",
        sa.Column("decision_id", sa.String, primary_key=True),
        sa.Column("session_id", sa.String, sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("state", sa.String, nullable=False),
        sa.Column("requested_action", sa.String, nullable=False),
        sa.Column("allowed", sa.Integer, nullable=False),
        sa.Column("reason_code", sa.String, nullable=False),
        sa.Column("policy_profile_id", sa.String, nullable=False),
        sa.Column("template_id", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_policy_decisions_session_id", "policy_decisions", ["session_id"])


def downgrade() -> None:
    op.drop_table("policy_decisions")
    op.drop_table("supervisor_interventions")
    op.drop_table("consent_artifacts")
    op.drop_table("audit_events")
    op.drop_table("validator_results")
    op.drop_table("model_proposals")
    op.drop_table("voice_turns")
    op.drop_table("sessions")
