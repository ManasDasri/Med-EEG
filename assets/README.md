Vendored browser dependencies for offline local-file use:

- React 18.2.0, MIT: https://github.com/facebook/react/blob/v18.2.0/LICENSE
- ReactDOM 18.2.0, MIT: same license
- Babel standalone 7.23.5, MIT: https://github.com/babel/babel/blob/v7.23.5/LICENSE

Files were retrieved from cdnjs version-specific paths. Retain license headers;
update intentionally and run the dashboard browser/validation checks. System
font fallbacks avoid an external font request. No API or network access is
required to read a local dashboard JSON file.
