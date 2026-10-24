HELP TEXT FORMAT
================

One .txt file per step of the program. The interface reads these files; the same text feeds the hover bubble, the footnote
line at the bottom, and the button that opens the file for the user to read. There is only one source, so they never disagree.

Each entry:

    [setting_key]
    short:  One sentence. Shown in the bubble and in the footnote line.
    plain:  A few plain words for someone new to this. Shown by default.
    detail: The technical wording. Shown when the user switches on show-details.
    see:    Where the design says more (a master section, for example).

Rules:
  * One entry per setting or tool. The key is the same word the program uses for the setting.
  * short, plain and detail are each written on one line (or continued on lines that start with two spaces).
  * No number is written here from memory: say "measured at start" and let the program show the value.
  * Blank lines between entries. Lines starting with # are comments.
