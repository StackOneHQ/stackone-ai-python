# Publishing stackone-ai

Release Please prepares the version and changelog in a release pull request.
Merging that release PR into `main` creates the GitHub release, tests and builds
its exact commit, then publishes the tested wheel and source archive to PyPI.
Normal development PRs do not publish. Changes to the release workflow run the
same build checks on the PR without access to publishing credentials.

## Configure PyPI before merging the publishing migration

A PyPI project owner must open **Manage → Publishing → Add a new publisher → GitHub**
for `stackone-ai` and register these values:

| Field | Value |
| --- | --- |
| Owner | `StackOneHQ` |
| Repository | `stackone-ai-python` |
| Workflow filename | `release.yaml` |
| Environment | `pypi` |

Use the filename alone, without `.github/workflows/`. Keep the environment set to
`pypi`, even though PyPI labels that field optional.

In GitHub repository settings, the `pypi` environment must allow deployment from
the **main branch only**, with no tag rules. This prevents an unreviewed branch
from replacing the publishing workflow and using the project's PyPI identity.
Keep any existing reviewer requirements.

Publishing uses GitHub's short-lived identity, called **Trusted Publishing** by
PyPI. No new PyPI API token or GitHub secret is needed. The upload job has no
source checkout and installs no package dependencies. Only that job can request
the identity needed to upload, and it downloads artifacts from its own workflow run.

## Release and verification

Review and merge the release PR, then check the **Release Please** workflow.
The source commit must be on `main`, and its package version must match the
GitHub release. A failed test or build prevents uploading. Confirm both files
appear on PyPI after the publish job succeeds.

Do not merge this migration until the PyPI publisher is configured. Keep the
existing token until the first successful publish through Trusted Publishing.
Then remove the unused `PYPI_API_TOKEN` secret and revoke its PyPI token after
checking that no other workflow uses it.
