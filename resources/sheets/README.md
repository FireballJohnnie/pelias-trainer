# Sheet templates

Three sheets, each shipped with erasable rows marked `EXAMPLE` in the `status` column (master 3.3, U7 and U8). A row still marked `EXAMPLE` produces a warning and never becomes a training row. Rows added by an import start as `DRAFT` and never train until the user accepts or edits them.

| Sheet | Columns |
|---|---|
| `persona.csv` | `key, value, status` |
| `parameters.csv` | `id, name, tier, description, answer_kind (code / text / none), answer, language, depends_on, status` |
| `clients.csv` | `client_name, system_prompt_style, system_prompt_text, roles_used, tool_schema, returns_reasoning (yes / no / unknown), source, status` |

`source` in `clients.csv` is one of `CAPTURED` (a real request someone logged), `PUBLISHED` (the client's documentation, link kept), `APPROXIMATION` (written by hand) or `UNVERIFIED`. The example rows are all `APPROXIMATION`; no product-named entry exists until its published request format has been looked up.

What a blank `answer` does is an open decision (master 16.7, decision 8).
