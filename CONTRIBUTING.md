# Contributing to Numba-CUDA

If you are interested in contributing to Numba-CUDA, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/numba-cuda/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The Numba-CUDA team will evaluate the issues and triage them. If you
      believe the issue needs priority attention comment on the issue to notify
      the team.
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

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA/numba-cuda/blob/main/README.md)
    to learn how to setup the development environment
2. Find an issue to work on.
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/NVIDIA/numba-cuda/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a Numba-CUDA developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

## Installation of Development Dependencies

There are some dependencies that are required for developing Numba-CUDA, but are not required for installation or distribution.
These dependencies are listed under the `test-cu11` and `test-cu12` optional dependency groups in our project configuration.

To install Numba-CUDA for development, run this in the root of the repository:

```shell
pip install -e ".[test-cu11]"
```
or
```sh
pip install -e ".[test-cu12]"
```

Numba-CUDA uses [`pre-commit`](https://pre-commit.com/) to run a number of style
checks in CI. To ensure your contribution will pass the checks, you can also set
up pre-commit locally to run the checks prior to committing.

```shell
# This will run a small set of checks every time you commit.
pre-commit install
```

To run them manually without committing, run this in the root of the repository:
```shell
pre-commit run --all-files
```

## Releases

The release process for Numba-CUDA involves the following steps:

- Open a PR to update `numba_cuda/VERSION` to the desired version.
- Generate a short changelog with `git log v<PREVIOUS_VERSION>..HEAD --oneline --pretty=format:"- %s"`
- Put the changelog in the version update PR description.
- Once `main` is updated, tag the release:
```
git checkout main && git pull
git tag -a v<VERSION>
```
- For the tag annotation, paste the same changelog as above, like this:
```
v<VERSION>

- ... (bullet points on release items)
```
- Push the tag:
```
git push git@github.com:NVIDIA/numba-cuda.git v<VERSION>
```

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
