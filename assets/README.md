# Instance Physics Ontology v1

This package contains field-level ontology JSON files for the first-stage visual-physical labeling pipeline.

Recommended usage:
1. Use the field JSON files to assemble prompts for VLM/LLM labeling.
2. Store actual instance annotations using `instance_record.template.json` as the record structure.
3. Use `ontology_manifest.json` to track which fields are active in this schema version.

Notes:
- Field files define ontology and annotation rules.
- Instance records should remain one JSON object per instance.
- `appearance_material_topk` encodes interaction-relevant material prototypes, not true composition ratios.
