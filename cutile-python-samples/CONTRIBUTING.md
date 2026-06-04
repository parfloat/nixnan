<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to cuTile

If you are interested in contributing to cuTile, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/nvidia/cutile-python/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The cuTile team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

You are welcome to make a pull request from your fork. Please ensure your change
meets the quality standard of the codebase.

Due to the limitation of Github CI, the main development is still happening in our
internal repository. We will provide a minimal CI in Github for you to iterate
on your changes. Once your PR passes the Github CI,
a reviewer will review the code. If it looks good, the reviewer will
test your change in our internal repository to ensure the change
passes the full CI check. Once it is merged into internal repository,
the change will be synced to the public Github repository and the PR will be marked
as "resolved" and closed.


### Your first issue

1. Read the project's [README.md](https://github.com/nvidia/cutile-python/blob/main/README.md)
    to learn how to setup the development environment.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/nvidia/cutile-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/nvidia/cutile-python/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/nvidia/cutile-python/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a cuTile developer will merge your pull request.

If you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Managing PR labels

Each PR must be labeled according to whether it is a "breaking" or
"non-breaking" change (using Github labels). This is used to highlight changes
that users should know about when upgrading.

For cuTile, a "breaking" change is one that modifies the public, non-experimental, Python API in a
non-backward-compatible way.

Additional labels must be applied to indicate whether the change is a feature,
improvement, bugfix, or documentation change.

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project board](https://github.com/orgs/nvidia/projects/99).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where cuTile developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Branches and Versions

### Branch naming

Branches used to create PRs should have a name of the form `<username>/<type>-<name>`
which conforms to the following conventions:
- Type:
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.


### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license. Signing off your commit means you accept the terms of the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
