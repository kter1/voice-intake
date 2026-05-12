/**
 * Verify TEMPLATE_CONTENT covers all backend template IDs from DEFAULT_TEMPLATE_BUNDLE.
 *
 * When a backend template ID is missing from this map, DemoPage renders [template_id]
 * as raw text to the user instead of the human-readable prompt. This test guards
 * against that regression whenever templates.py grows or is changed.
 */
import { describe, expect, it } from "vitest";
import { TEMPLATE_CONTENT } from "../pages/DemoPage";

// All template_id values from DEFAULT_TEMPLATE_BUNDLE in src/voice_intake/templates.py.
// Keep in sync with that file when templates are added or removed.
const BACKEND_TEMPLATE_IDS = [
  "opening_disclosure",
  "demo_capabilities_prompt",
  "recording_consent_prompt",
  "ai_assistance_consent_prompt",
  "collect_field_prompt",
  "readback_confirmation",
  "identity_threshold_denial",
  "hold_message",
  "emergency_redirect",
  "closing_prompt",
  "appointment_reason_prompt",
  "patient_name_prompt",
  "dob_prompt",
  "dob_clarification",
  "insurance_prompt",
  "insurance_in_network",
  "insurance_out_of_network",
  "insurance_unknown",
  "insurance_not_provided",
  "appointment_scheduled",
  "appointment_request_created",
  "human_handoff",
  "cannot_schedule_without_identity",
  "field_explanation_for_reason",
  "field_explanation_for_name",
  "field_explanation_for_dob",
  "field_explanation_for_insurance",
] as const;

describe("TEMPLATE_CONTENT - frontend render map coverage", () => {
  it.each(BACKEND_TEMPLATE_IDS)(
    "covers backend template '%s' with non-empty text",
    (id) => {
      expect(TEMPLATE_CONTENT).toHaveProperty(id);
      expect(TEMPLATE_CONTENT[id].trim().length).toBeGreaterThan(0);
    },
  );

  it("does not fall back to [template_id] for consent templates", () => {
    expect(TEMPLATE_CONTENT["recording_consent_prompt"]).not.toMatch(
      /^\[.*\]$/,
    );
    expect(TEMPLATE_CONTENT["ai_assistance_consent_prompt"]).not.toMatch(
      /^\[.*\]$/,
    );
    expect(TEMPLATE_CONTENT["closing_prompt"]).not.toMatch(/^\[.*\]$/);
  });

  it("consent templates match exact backend text", () => {
    expect(TEMPLATE_CONTENT["recording_consent_prompt"]).toBe(
      "Do you consent to call recording where required by law?",
    );
    expect(TEMPLATE_CONTENT["ai_assistance_consent_prompt"]).toBe(
      "Do you consent to speaking with an AI assistant supervised by staff?",
    );
    expect(TEMPLATE_CONTENT["closing_prompt"]).toBe(
      "Thank you. Your intake has been recorded for staff follow-up.",
    );
  });
});
