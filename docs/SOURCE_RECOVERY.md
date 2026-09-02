# Source Recovery Record

The v2.0 PDF contains a numbered source distribution. Literal layout extraction
recovered 48 Python files: 35 compiled immediately and 13 initially failed
because page boundaries removed or added common indentation.

For v2.1 repository assembly, those 13 files were repaired by restoring the
surrounding Python block indentation. The recovery did not replace expected
values, relax assertions or alter scientific formulas to make tests pass.
After repair:

- every source, test and example file passes `compileall`;
- the registry loads all 33 models with zero structural-audit problems;
- all 118 baseline tests pass;
- the source remains classified `RECOVERED_AND_RERUN`, not identical to an
  unavailable pre-PDF canonical repository.

The supplied baseline PDF SHA-256 is:

```text
90c4fdad85b4a08750a80bca33f8bf75beccb9af2b1782233fdf9b38e88ccf5f
```

This digest identifies the supplied bytes. A different digest printed inside
the PDF does not match those supplied bytes and is not represented here as a
verified external checksum.

