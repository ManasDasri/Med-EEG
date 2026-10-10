# Understanding `inspect_dataset.py`: a beginner's walkthrough

This guide explains [the complete inspection script](../src/inspect_dataset.py), including the Python concepts behind it. You do not need prior experience with pandas or EEG processing.

## 1. What problem does this script solve?

Before processing EEG recordings, we need to know what files we have and how their metadata is named.

`inspect_dataset.py` looks through a downloaded dataset and prints:

- The participant folders it finds.
- The column names and first few rows of the participant table.
- The event labels used across the participants' event tables.
- The channel names from one example recording.
- Warnings and suggestions for manual inspection.

It does **not** load EEG waveforms, clean signals, calculate features, train a model, or download data.

Think of it as checking the labels and contents list on a collection of boxes before unpacking them.

Its place in the project is approximately:

```text
Downloaded dataset
        |
        v
inspect_dataset.py   Look at file organization and metadata
        |
        v
preprocess.py        Read and clean EEG; create epochs
        |
        v
extract_features.py  Turn those epochs into numeric features
```

The inspection script mostly prints information for a human. It is not a complete automated validator.

## 2. The files and EEG terms you need to know

### Dataset and BIDS

A **dataset** is the collection of recordings and associated information.

**BIDS** stands for Brain Imaging Data Structure. For this script, the relevant idea is that files and folders follow naming conventions.

A simplified layout might look like this:

```text
data/ds001787/
├── participants.tsv
├── sub-01/
│   └── ses-01/
│       └── eeg/
│           ├── sub-01_ses-01_task-example_events.tsv
│           ├── sub-01_ses-01_task-example_channels.tsv
│           └── ... EEG recording files ...
└── sub-02/
    └── ...
```

This is an illustration, not a listing of files verified on your machine.

| Term | Meaning in this project |
|---|---|
| Subject / participant | A person who took part in the experiment |
| Session | A recording visit or session belonging to a participant |
| Channel | An electrode or another recorded sensor signal |
| Event | Something recorded as happening at a particular time |
| Probe | A prompt asking the participant to report their state |
| Rating | A participant's answer, such as their concentration rating |
| Metadata | Information describing the recordings, rather than the waveform samples themselves |

The script searches inside session folders, but it does not explicitly count or validate sessions.

### What is a TSV file?

TSV means **tab-separated values**. It is a text file containing a table.

For example, a participant table might contain:

```text
participant_id    group
sub-01            experienced
sub-02            novice
```

In a real TSV, the column separators are tab characters. The spacing above is just for readability.

The files inspected are:

| Filename or pattern | Purpose |
|---|---|
| `participants.tsv` | Participant-level information |
| `*_events.tsv` | Event information for recordings |
| `*_channels.tsv` | Channel information for recordings |

The `*` means “any sequence of characters.” Thus `*_events.tsv` matches filenames ending in `_events.tsv`.

## 3. A few Python basics before reading the file

### Variables and assignment

```python
n_events_files = 0
```

This assigns the integer `0` to the name `n_events_files`.

Python uses `=` for assignment and `==` for equality comparison:

```python
count = 24       # Give count a value.
count == 24     # Ask whether count equals 24.
count != 24     # Ask whether count does not equal 24.
```

### Functions and calls

A function is a named block of reusable instructions:

```python
def greet(name):
    print(name)
```

`def` defines the function. `name` is a **parameter**: a name for information the function receives.

```python
greet("Asha")
```

This **calls** the function. `"Asha"` is the **argument** supplied to the parameter.

Defining a function does not execute its body. The body runs when the function is called.

### Indentation defines blocks

```python
if count != 24:
    print("Unexpected count")
print("Finished checking")
```

The indented line belongs to the `if`. The last line does not, so it runs regardless of the condition.

Colons introduce blocks after constructs such as `def`, `if`, `else`, and `for`. This project uses four spaces for indentation.

### Common value types

| Type | Example | Meaning |
|---|---|---|
| `str` | `"sub-01"` | Text |
| `int` | `24` | Whole number |
| `bool` | `True`, `False` | A true/false value |
| `list` | `["sub-01", "sub-02"]` | An ordered collection |
| `set` | `{"stimulus", "response"}` | A collection of distinct values |
| `None` | `None` | No value / no result |
| `Path` | `Path("data/ds001787")` | An object representing a filesystem path |
| `DataFrame` | A pandas table | Rows and named columns |

A **method** is a function accessed through an object, such as `df.head()` or `path.exists()`. An **attribute**, such as `df.columns` or `path.name`, is accessed without calling it with parentheses.

## 4. The opening docstring

The file begins with text surrounded by triple quotes:

```python
"""Step 1: point at the downloaded BIDS dataset and sanity-check its layout.
...
"""
```

This is the module's **docstring**: documentation stored as a string at the start of the file. A module is a Python file that can be run or imported.

A docstring does not automatically print to the terminal. It explains the intended purpose and provides an example command.

The opening description is broader than the actual checks: the code prints useful metadata, but it does not verify every session, channel, or probe structure.

## 5. Imports: bringing in existing tools

```python
from __future__ import annotations
```

This postpones the evaluation of type annotations. You can understand the rest of the file without studying that mechanism: it affects how Python handles the type descriptions attached to functions and variables.

```python
import argparse
import sys
from pathlib import Path

import pandas as pd
```

| Import | Why this script uses it |
|---|---|
| `argparse` | Read options supplied in the terminal |
| `sys` | Access Python's module search path |
| `Path` from `pathlib` | Work with files and directories |
| `pandas`, renamed locally to `pd` | Read and inspect tabular files |

`argparse`, `sys`, and `pathlib` come with Python. pandas is an external dependency that must be installed in your Python environment.

`as pd` creates an alias. That is why the code writes `pd.read_csv(...)` instead of `pandas.read_csv(...)`.

### Making the local configuration importable

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROBE_EVENT_KEYWORDS, Paths  # noqa: E402
```

The first line is easier to understand from the inside outward:

1. `__file__` identifies the current Python file.
2. `Path(__file__)` represents its location as a path object.
3. `.resolve()` obtains an absolute path and resolves symbolic links.
4. `.parent` selects the containing directory: here, `src`.
5. `str(...)` converts the path to text.
6. `sys.path.insert(0, ...)` places that directory first in Python's list of places to search for imported modules.

The next line imports two names from [config.py](../src/config.py):

- `PROBE_EVENT_KEYWORDS`: words used to recognize possible probe labels.
- `Paths`: a helper for constructing project paths.

`# noqa: E402` is a comment for linting tools. It suppresses their complaint about an import appearing after executable code. It does not change Python's runtime behavior.

## 6. `find_subjects()`: discover participant folders

```python
def find_subjects(bids_root: Path) -> list[str]:
    return sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
```

### The function signature

`bids_root: Path` says the function expects a path object.

`-> list[str]` says it returns a list of strings.

These are **type hints**. They help readers, editors, and type-checking tools; they do not automatically enforce the types when you call the function.

### What the body does

```python
bids_root.glob("sub-*")
```

This finds immediate children of the dataset directory whose names begin with `sub-`.

The expression inside `sorted()` is a **generator expression**:

```python
p.name for p in bids_root.glob("sub-*") if p.is_dir()
```

It produces each matching folder's name, skipping matching items that are not directories. `p` is just a temporary variable for the current path.

`sorted()` consumes those names and returns a sorted list. `return` sends that result back to the caller and exits the function.

An expanded equivalent is:

```python
def find_subjects(bids_root):
    names = []
    for p in bids_root.glob("sub-*"):
        if p.is_dir():
            names.append(p.name)
    return sorted(names)
```

`append()` adds one item to the end of a list.

Example:

```text
Matching folders: sub-03, sub-01, sub-02
Returned list:   ["sub-01", "sub-02", "sub-03"]
```

Sorting is by text, not by interpreting the suffix as a number. Consistently zero-padded identifiers make that ordering easier to read.

## 7. `inspect_participants_tsv()`: preview participant metadata

```python
def inspect_participants_tsv(bids_root: Path) -> pd.DataFrame | None:
```

The return type means “a pandas DataFrame **or** `None`.” The vertical bar represents a union of possible types.

### Build the path

```python
f = bids_root / "participants.tsv"
```

For a `Path` object, `/` joins path components. It is not doing division here.

If `bids_root` is `Path("data/ds001787")`, then `f` represents `data/ds001787/participants.tsv`.

### Handle a missing file

```python
if not f.exists():
    print(f"[warn] no participants.tsv at {f} -- group labels (expert/"
          f"novice) will need to come from elsewhere.")
    return None
```

`.exists()` returns `True` if the path exists. `not` reverses the Boolean, so the block runs when the path does not exist.

The leading `f` before a quoted string makes it an **f-string**. Expressions inside `{...}` are inserted into the text. Here, `{f}` displays the file path. The `f` prefix and the variable named `f` happen to use the same letter, but serve unrelated purposes.

The two adjacent string literals are automatically combined. Splitting them over lines keeps the source readable.

`return None` is an **early return**: stop this function without trying to read a missing file. The rest of the script can continue.

### Read and print the table

```python
df = pd.read_csv(f, sep="\t")
print(f"[ok] participants.tsv columns: {list(df.columns)}")
print(df.head())
return df
```

- `read_csv()` reads a delimited text table despite its CSV-oriented name.
- `sep="\t"` sets the separator to a tab. `\t` is Python's escape sequence for a tab character.
- `df` is a conventional abbreviation for DataFrame.
- `df.columns` contains the column labels.
- `list(...)` turns those labels into a regular Python list for display.
- `df.head()` returns the first five rows by default.
- `print(...)` writes a representation to the terminal.

The function returns the entire table, not just its first five rows. However, `main()` calls this function without storing the result, so only its printing is used here.

**Limit:** this function does not check whether a `group` column exists or whether its values are valid.

## 8. `inspect_events()`: collect and inspect event labels

```python
def inspect_events(bids_root: Path, subjects: list[str]) -> None:
```

This receives a dataset path and the list of participant folder names. `-> None` indicates that it produces no useful return value; its purpose is to print information.

### Initialize the collection and counter

```python
print("\n--- scanning *_events.tsv for probe/rating trial_type values ---")
seen_types: set[str] = set()
n_events_files = 0
```

`\n` inserts a newline, leaving a visual gap before the message.

`set()` creates an empty set. Sets keep only distinct elements: adding `"response"` repeatedly still leaves one `"response"` entry.

The annotation `set[str]` describes a set intended to contain strings. Again, it does not automatically convert incoming values to strings.

`n_events_files` counts how many matching files are visited, not how many events or probes exist.

### Search every subject recursively

```python
for sub in subjects:
    for events_file in (bids_root / sub).rglob("*_events.tsv"):
        n_events_files += 1
        df = pd.read_csv(events_file, sep="\t")
```

These are **nested loops**. For each participant, the inner loop processes their matching files.

`rglob()` searches recursively through nested folders. Unlike the earlier `glob("sub-*")`, this can find files beneath session and EEG directories.

`n_events_files += 1` is shorthand for:

```python
n_events_files = n_events_files + 1
```

### Collect distinct `trial_type` values

```python
if "trial_type" in df.columns:
    seen_types.update(df["trial_type"].dropna().unique().tolist())
```

`in` tests membership: does the table contain a column named `trial_type`?

The chained operations mean:

| Expression | Result |
|---|---|
| `df["trial_type"]` | Select that column as a pandas Series |
| `.dropna()` | Remove missing entries from the selected data |
| `.unique()` | Get the distinct values |
| `.tolist()` | Convert the result to a Python list |
| `seen_types.update(...)` | Add all those values to the set |

A **Series** is pandas' one-dimensional labeled data structure; selecting a single DataFrame column normally gives you one.

These operations are used to build the label collection. They do not write changes back to the TSV file.

Example:

```text
File A trial_type: stimulus, response, response, missing
File B trial_type: stimulus, response

Combined distinct labels: stimulus, response
```

If a file lacks `trial_type`, it still contributes to the file count, but contributes no labels. The script does not issue a specific warning for that file.

### Print the results

```python
print(f"[ok] found {n_events_files} events.tsv files across "
      f"{len(subjects)} subjects")
print(f"[ok] unique trial_type values seen: {sorted(seen_types)}")
```

`len(subjects)` is the number of participant names in the list.

`sorted(seen_types)` produces a sorted list for readable output. Sets themselves do not provide a meaningful display order.

### Search for probe-related words

```python
probe_like = [t for t in seen_types
              if any(k in str(t).lower() for k in PROBE_EVENT_KEYWORDS)]
```

This is a **list comprehension**: build a list containing the labels that pass a condition.

The configured keywords are:

```python
("probe", "concentration", "mind_wandering", "rating")
```

This collection is a **tuple**, an ordered collection whose entries cannot be reassigned in place. Uppercase names such as `PROBE_EVENT_KEYWORDS` conventionally indicate configuration constants; Python does not enforce immutability of the name.

For each label `t`:

1. `str(t)` converts it to text.
2. `.lower()` converts that text to lowercase.
3. `k in ...` checks whether keyword `k` appears anywhere inside the text.
4. `any(...)` returns `True` if at least one keyword matches.
5. The original label `t` is included if the condition is true.

For example:

| Event label | Match? | Reason |
|---|---|---|
| `ConcentrationRating` | Yes | Its lowercase form contains `concentration` and `rating` |
| `probe_start` | Yes | Contains `probe` |
| `stimulus` | No | Contains none of the configured keywords |
| `response` | No | Contains none of the configured keywords |

This is a substring check, not an interpretation of the event's scientific meaning.

An expanded equivalent is:

```python
probe_like = []
for t in seen_types:
    for k in PROBE_EVENT_KEYWORDS:
        if k in str(t).lower():
            probe_like.append(t)
            break
```

`break` exits the inner loop once a match is found, so the same label is not added again for another matching keyword.

### Print a match or a suggestion

```python
if probe_like:
    print(f"[ok] probe-like event labels: {probe_like}")
else:
    print(...)
```

Python treats an empty list as false and a nonempty list as true. This is called **truthiness**. Here, `if probe_like:` means “if the list contains at least one item.”

The `else` prints advice to inspect an event file manually. It does not stop execution.

### Important: the keyword check is behind the current preprocessing logic

The current [preprocess.py](../src/preprocess.py) already recognizes a probe using:

```python
trial_type == "stimulus"
value == 128
```

It then decodes following `response` rows. This inspection function never examines `value`, so it can print `[action needed]` even when preprocessing supports the event format.

Changing `PROBE_EVENT_KEYWORDS` changes this inspection's keyword check; it does not replace the actual probe-decoding logic in preprocessing. A label matching a keyword also does not prove that its responses can be decoded correctly.

## 9. `inspect_channels()`: show one example channel list

```python
def inspect_channels(bids_root: Path, subjects: list[str]) -> None:
    if not subjects:
        return
```

If there are no participant folders, the function exits. A bare `return` returns `None`.

This prevents the next line from trying to select an item from an empty list:

```python
sample = subjects[0]
```

Python uses **zero-based indexing**: `[0]` is the first item, `[1]` is the second, and so on.

Because `find_subjects()` sorted the names, `sample` is the first participant in that sorted order.

### Find channel tables for that participant

```python
ch_files = list((bids_root / sample).rglob("*_channels.tsv"))
if not ch_files:
    print(f"[warn] no *_channels.tsv found for {sample}")
    return
```

`rglob()` yields paths as it searches. Wrapping it in `list()` collects the paths so the code can check whether the collection is empty and select its first item.

### Read the first matching file

```python
df = pd.read_csv(ch_files[0], sep="\t")
print(f"\n--- channel names for {sample} ({ch_files[0].name}) ---")
```

`ch_files[0]` is the first matching file. The code does not explicitly sort these paths, so it does not guarantee a particular session is selected.

`.name` extracts just the filename for the heading.

### Print the channel names

```python
print(df["name"].tolist() if "name" in df.columns else df.iloc[:, 0].tolist())
```

This uses a **conditional expression**:

```python
value_if_true if condition else value_if_false
```

It means:

```python
if "name" in df.columns:
    names = df["name"].tolist()
else:
    names = df.iloc[:, 0].tolist()
print(names)
```

`iloc` selects pandas data by integer position. In `df.iloc[:, 0]`:

- `:` means all rows.
- `0` means the first column.
- The comma separates the row selection from the column selection.

The fallback assumes the first column is useful to display. It does not verify that the column actually contains channel names.

### The manual-check message

The function prints a request to confirm that the configured frontal and posterior electrode names exist.

This message is unconditional after a channel file is read. The function does **not** compare the names itself.

Also, the current preprocessing code renames BioSemi channel labels, such as `A1` and `B1`, into standard electrode names using the mapping in `config.py`. Therefore, the raw channel table does not necessarily need to contain the final names such as `Fz` or `Pz` literally.

This function checks only one file from one participant. It does not establish that all recordings have consistent channels.

## 10. `main()`: connect the pieces

### Create the command-line parser

```python
parser = argparse.ArgumentParser()
parser.add_argument("--bids_root", required=True, type=Path,
                     help="Path to the downloaded ds001787 / BIDS_EEG_"
                          "meditation_experiment folder")
args = parser.parse_args()
```

`ArgumentParser()` creates an object that knows how to interpret command-line arguments.

`add_argument()` defines one option:

| Part | Meaning |
|---|---|
| `"--bids_root"` | The option's name in the terminal |
| `required=True` | The user must provide it |
| `type=Path` | Convert the supplied text into a `Path` object |
| `help=...` | Description shown in command-line help |

Values written as `name=value` inside a function call are **keyword arguments**. They explicitly name the parameter being supplied.

`parse_args()` reads the arguments and stores them in an object. `args.bids_root` accesses the parsed path.

From the project root:

```bash
python src/inspect_dataset.py --bids_root data/ds001787
```

The components mean:

```text
python                      Run the Python interpreter
src/inspect_dataset.py      Execute this script
--bids_root                 Supply the dataset-location option
data/ds001787               Use this directory as its value
```

Relative paths are interpreted relative to the terminal's current working directory, not automatically relative to the script.

For a path containing spaces, quote it:

```bash
python src/inspect_dataset.py --bids_root "/path/to/my dataset"
```

To display help:

```bash
python src/inspect_dataset.py --help
```

The script's imports still need to succeed before argument parsing can show the help.

### Stop if the root path does not exist

```python
if not args.bids_root.exists():
    raise SystemExit(...)
```

`raise` raises an exception. `SystemExit` is the exception used here to end the program with a message and an unsuccessful exit status.

This is different from printing `[warn]`: a printed warning alone does not stop execution.

The check tests existence. It does not explicitly verify that the path is a directory containing a complete BIDS dataset.

### Construct shared paths

```python
paths = Paths.from_root(args.bids_root)
```

`Paths` is a dataclass defined in `config.py`: a class used to hold related values. A **class** defines a kind of object; here an instance stores `bids_root` and `derivatives`.

`from_root()` is a class method. Its relevant implementation is:

```python
root = Path(bids_root)
deriv = root / "derivatives" / "neurodial"
deriv.mkdir(parents=True, exist_ok=True)
return cls(bids_root=root, derivatives=deriv)
```

- `parents=True` allows missing parent directories to be created.
- `exist_ok=True` allows the directory to exist already.
- `cls(...)` constructs an instance of the class.

**Side effect:** running the inspector can create `derivatives/neurodial/`. It does not write a report there or alter the raw recordings.

### Count the participants

```python
subjects = find_subjects(paths.bids_root)
print(f"[ok] found {len(subjects)} subjects: {subjects}")
if len(subjects) != 24:
    print(...)
```

The number `24` is a project-specific expectation, not a general BIDS rule. A different number produces a warning, then the script continues.

### Run the inspections in order

```python
inspect_participants_tsv(paths.bids_root)
inspect_events(paths.bids_root, subjects)
inspect_channels(paths.bids_root, subjects)
```

These calls run sequentially. Each must finish before the next begins.

Finally, the script prints advice about reviewing any action items before preprocessing. Reaching that final message means execution reached the end; it does not mean every check passed.

## 11. Why is `main()` called at the bottom?

```python
if __name__ == "__main__":
    main()
```

Python gives each module a special `__name__` value.

- When you run the file directly, its name is `"__main__"`, so `main()` runs.
- When another module imports the file, its name is the module name, so this block does not call `main()`.

This allows other code to import functions such as `find_subjects()` without automatically starting the full inspection.

Importing the file still executes its top-level imports and the `sys.path` modification. The guard prevents the `main()` call, not every top-level operation.

## 12. The complete execution order

The source places helper functions above `main()`, but their bodies do not execute merely because Python encounters their definitions.

When run directly, the sequence is:

```text
Load imports and configuration
        |
Define the functions
        |
Check __name__ and call main()
        |
Parse --bids_root
        |
Check root exists -------- no --------> Exit with a message
        |
       yes
        |
Construct Paths; ensure derivatives/neurodial exists
        |
Find participant folders; print count and any count warning
        |
Preview participants.tsv, or warn if absent
        |
Read event tables and inspect their labels
        |
Read one channel table, if available
        |
Print final advice
```

File-reading or other unexpected errors can interrupt this sequence. There is no general `try`/`except` error handler around these operations.

## 13. Understanding the messages and limitations

The prefixes are ordinary text chosen by the author. They are not a special Python validation system.

| Message or behavior | What it establishes | What it does not establish |
|---|---|---|
| `[ok] found ... subjects` | Matching directories were counted | Every participant has complete recordings |
| Participant columns and preview | The participant table was read | Group labels are present and correct |
| `[ok] found ... events.tsv files` | Matching files were counted and read | Every file contains valid event columns |
| Unique `trial_type` list | These labels occurred in the scanned tables | Probes and answers have been decoded |
| Probe-like label match | A keyword appeared in a label | The event is definitely a valid probe |
| Channel list | One table's names, or first column, were displayed | All recordings have the required channels |
| `[action needed]` for channels | The script printed its standard reminder | A missing electrode was actually detected |
| Final `Done` message | Execution reached the final print | The dataset passed comprehensive validation |

### Situations worth recognizing

- **Missing `participants.tsv`:** prints a warning and continues.
- **No matching subjects:** prints a count warning; the channel inspection returns early.
- **Missing `trial_type`:** that event file is counted but adds no labels.
- **Generic labels such as `stimulus` and `response`:** may produce a keyword warning even though current preprocessing supports them.
- **No channel table for the first participant:** prints a warning; it does not try another participant.
- **Unreadable or malformed files:** can raise errors and stop the script.
- **Unexpected non-text event labels:** the collection is annotated as strings but not enforced; incompatible mixed types can cause sorting to fail.

Do not automatically change the preprocessing configuration just because a generic action message appears. Compare the displayed metadata with the current decoding and channel-renaming logic first.

## 14. Small Python examples you can experiment with

These examples operate on in-memory values and do not change dataset files.

### Lists, indexing, and length

```python
subjects = ["sub-01", "sub-02", "sub-03"]
print(subjects[0])   # sub-01
print(len(subjects)) # 3
```

### Sets remove duplicates

```python
labels = set()
labels.update(["stimulus", "response", "response"])
print(sorted(labels))  # ['response', 'stimulus']
```

### Keyword matching

```python
label = "ConcentrationRating"
keywords = ("probe", "concentration", "rating")
print(any(word in label.lower() for word in keywords))  # True
```

### Path objects describe locations

```python
from pathlib import Path

root = Path("data/ds001787")
file = root / "participants.tsv"
print(file.name)    # participants.tsv
print(file.parent)  # data/ds001787
```

Constructing these path objects does not create files or folders. An operation such as `mkdir()` is needed to create a directory.

### A small pandas table

```python
import pandas as pd

df = pd.DataFrame({
    "trial_type": ["stimulus", "response", None, "response"]
})

print(df["trial_type"].dropna().unique().tolist())
# ['stimulus', 'response']
```

The `{...}` passed to `DataFrame` is a **dictionary**: it maps the column name to a list of values.

## 15. Quick syntax reference

| Syntax | Meaning |
|---|---|
| `# comment` | A comment ignored during normal execution |
| `"""text"""` | A multiline string; used at the start of this file as a docstring |
| `def name(...):` | Define a function |
| `return value` | Exit the function and give a value to its caller |
| `-> None` | Type hint indicating no useful return value |
| `list[str]` | Type hint for a list of strings |
| `A \| None` | Type hint allowing either an `A` value or `None` |
| `object.attribute` | Access an attribute |
| `object.method()` | Call a method |
| `if ...:` / `else:` | Choose which block to execute |
| `for item in items:` | Repeat for each item |
| `not value` | Negate a truth value |
| `x in collection` | Test membership; for strings, test whether a substring occurs |
| `items[0]` | Select the first item |
| `df.iloc[:, 0]` | Select all rows of the first column by position |
| `f"Found {n}"` | Format text using an expression's value |
| `\n` / `\t` | Newline / tab escape sequences |
| `[x for x in items if condition]` | Build a filtered list |
| `(x for x in items)` | Produce values with a generator expression |
| `any(values)` | True if at least one value is truthy |
| `raise SystemExit(message)` | Terminate this command-line program with a message |

When rereading the actual script, start with `main()` to see the sequence, then follow each function call. That order makes it easier to connect the individual Python expressions to the job they perform.
