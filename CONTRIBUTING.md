<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to the Large Language Model Training Playbook

The Large Language Model Training Playbook is a living document. We anticipate regular improvements, so please please watch the repository to be notified about these.

Everyone is welcome to contribute, and we value everybody's contribution. New content writing
contributions are not the only way to help. Answering questions in issues, helping
others in pull-request, and improving the existing writing are also often valuable.

Though, please don't file a pull request without first coordinating via the issue system (see below) as (1) it might be content that goes beyond what the playbook is intended to cover or (2) someone else might already be working on this.

Feel also free to spread the word! You can reference the playbook in blog posts or shout out on Twitter every time if it has helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/large_language_model_training_playbook/blob/main/CODE_OF_CONDUCT.md).

**This guide was inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to the "Large Language Model Training Playbook":

* Propose a new section or propose to add more content to an existing section.
* Submit issues about inexatitude or clarity on current content.
* Read and comment on a pull request proposing new content or correcting the existing content.

If you don't know where to start, there might be special [Good First
Issue](https://github.com/huggingface/large_language_model_training_playbook/contribute) listing. It will give you a list of open issues that are beginner-friendly and help you start contributing to open-source. Just comment in the issue that you'd like to work on it. 

> All contributions are equally valuable to the community. 🥰

## Propose a new section and/or additional content

If you would like to add a new section or content to an existing section, please **open an issue first to discuss the matter** before creating a pull request.

Even though the project aim at integrating as much as possible inputs from any contributors, we don't garantee we'll accept all topics or contributions so it's always better to approval before starting to spend significant amount of time on a writing section.

## Submit issues about inexatitude or clarity on current content

When submitting an issue about inexatitude or clarity on current content please be careful about our
[code of conduct](https://github.com/huggingface/large_language_model_training_playbook/blob/main/CODE_OF_CONDUCT.md) as we prohibit some behaviors and type of communication. In particular we try to build a positive environment for our
community by being respectful of differing opinions, viewpoints, and experiences and giving and gracefully accepting constructive feedback. In a nutshell: don't forget there is a human just like you at the other side who has likely spend time and effort writing the content you are now commenting.

The repo maintainers will be very strict regarding any action they deem in violation of this Code of Conduct (see the [Enforcement Guidelines section of the Code of Conduct](https://github.com/huggingface/large_language_model_training_playbook/blob/main/CODE_OF_CONDUCT.md#Enforcement-Guidelines))

## Create a Pull Request

Before writing any section or content, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute to the
🤗 Large Language Model Training Playbook. While `git` is not the easiest tool to use, it has the greatest
manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/large_language_model_training_playbook) by
   clicking on the **[Fork](https://github.com/huggingface/large_language_model_training_playbook/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/large_language_model_training_playbook.git
   $ cd large_language_model_training_playbook
   $ git remote add upstream https://github.com/huggingface/large_language_model_training_playbook.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Write the content in your branch.

   You can now write the new content or the correction you wanted to submit.

   Once you're happy with your changes, add changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   $ git add modified_file.md
   $ git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

5. Now you can go to your fork of the repository on GitHub and click on **Pull request** to open a pull request. When you're ready, you can send your changes to the project maintainers for review.

6. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Develop on Windows

On Windows (unless you're working in [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) or WSL), you need to configure git to transform Windows `CRLF` line endings to Linux `LF` line endings:

```bash
git config core.autocrlf input
```

One way to run the `make` command on Windows is with MSYS2:

1. [Download MSYS2](https://www.msys2.org/), and we assume it's installed in `C:\msys64`.
2. Open the command line `C:\msys64\msys2.exe` (it should be available from the **Start** menu).
3. Run in the shell: `pacman -Syu` and install `make` with `pacman -S make`.
4. Add `C:\msys64\usr\bin` to your PATH environment variable.

You can now use `make` from any terminal (Powershell, cmd.exe, etc.)! 🎉

### Sync a forked repository with upstream main (the Hugging Face repository)

When updating the main branch of a forked repository, please follow these steps to avoid pinging the upstream repository which adds reference notes to each upstream PR, and sends unnecessary notifications to the developers involved in these PRs.

1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead, merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:

```bash
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```
