# Publishing Checklist

1. Run `pytest`
2. Update version in `reno/__init__.py`
3. Update changelog (`CHANGELOG.md`)
4. If new docs version, be sure to update `sphinx/source/_static/switcher.json`
5. Update docs as needed with `make html` inside `sphinx` dir
6. Add docs version folder with `make apply-docs`
7. Commit
8. Push to github
9. Publish to pypi (`make publish`)
10. Tag release on github
