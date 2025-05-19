# GitHub Repository: Useful Commands Cheat Sheet

## BASICS

```bash
git status
    Show changes in working directory (what's staged, unstaged, untracked)

git diff
    Show unstaged changes (line-by-line diff)

git add <file> / git add .
    Stage changes (prepare them for commit)

git reset <file>
    Unstage a file (but keep your changes)

git commit -m "message"
    Commit staged changes with a message

git push
    Push commits to GitHub (after upstream is set)

git pull
    Pull latest changes from GitHub

git log --oneline
    Compact history view of commits
```

---

## BRANCHING

```bash
git branch
    List all branches

git branch <name>
    Create a new branch

git checkout <name>
    Switch to another branch

git checkout -b <name>
    Create and switch to a new branch

git merge <branch>
    Merge another branch into your current one
```

---

## TAGS

```bash
git tag
    List tags

git tag v1.0
    Create tag v1.0 for current commit

git push origin v1.0
    Push a tag to GitHub
```

---

## REMOTE SETUP

```bash
git remote -v
    Show linked GitHub remotes

git remote add origin git@github.com:MartinTrappe/mpDPFT.git
    Link local repo to GitHub

git push -u origin master
    Push and set upstream (only needed once per branch)
```

---

## FILE TRACKING

```bash
git rm <file>
    Delete file and stop tracking it

git rm --cached <file>
    Stop tracking file but keep it on disk

git ls-files --others --ignored --exclude-standard
    List ignored (untracked) files
```

---

## SAFETY & CLEANUP

```bash
git log --all -- <file>
    Show all commits touching a file

git reset --hard HEAD
    Undo all local changes (dangerous!)

git clean -fd
    Remove all untracked files and folders
```


# Git Pull Request Cheat Sheet (command line. Alternatively: online at GitHub)

## 0. (Optional) Save or stash any in-progress work on master

```bash
git status
# – shows dirty files
```

**Option A: Commit your WIP**

```bash
git add .
git commit -m "WIP: save before merging contour"
```
- **remote:** none
- **local:** creates a new commit on `master`
- **wd:** now clean

**Option B: Stash your WIP**

```bash
git stash push -m "WIP before merging contour"
```
- **remote:** none
- **local:** writes your changes to `refs/stash`
- **wd:** resets tracked files to HEAD, but **untracked** files remain untouched

---

## 1. Fetch updates from origin

```bash
git fetch origin
```
- **remote:** Git contacts the origin server, sends you any new commits, trees, blobs, tags
- **local:** updates `refs/remotes/origin/*` (e.g. `origin/master`, `origin/contour`)
- **wd:** no change

---

## 2. (If needed) Create & switch to a local `contour` branch

```bash
git branch contour origin/contour
```
- **remote:** none
- **local:** creates `refs/heads/contour` → same commit as `origin/contour`
- **wd:** no change

```bash
git checkout contour
```
- **remote:** none
- **local:**
  - HEAD moves to `refs/heads/contour`
  - Index (staging area) reset to the tree of that commit
- **wd:**
  - **Tracked files** are overwritten, created, or deleted to match `contour`’s snapshot
  - **Untracked files** are left alone
  - If a tracked file would overwrite an untracked file, Git errors out

---

## 3. Update & switch back to `master`

```bash
git checkout master
```
- **remote:** none
- **local:**
  - HEAD moves to `refs/heads/master`
  - Index reset to master’s tree
- **wd:** tracked files updated to match your local `master`; untracked still untouched

```bash
git pull --ff-only origin master
```
- **remote:** fetches `origin/master` (if needed)
- **local:** fast-forwards `refs/heads/master` → `origin/master`
- **wd:** tracked files updated to match new master tip

---

## 4. Merge in `contour`’s work

```bash
git merge --no-ff contour
```
- **remote:** none
- **local:**
  - finds a merge-base, performs a three-way merge
  - writes new blobs/trees, and makes a merge commit on `refs/heads/master`
- **wd:**
  - applies merged content
  - if conflicts occur, conflict markers are left in affected files

---

## 5. Resolve conflicts & finalize

```bash
git add .
git commit
```
- **remote:** none
- **local:** completes the merge commit
- **wd:** clean, all conflicts resolved

---

## 6. Push your updated `master` back to origin

```bash
git push origin master
```
- **remote:** receives your merge commit & any new objects, updates `origin/master`
- **local:** `refs/heads/master` already at tip
- **wd:** unchanged

---

## Quick “at-a-glance” summary

| Command                           | Remote                                     | Local Metadata                             | Working Directory                                           |
|-----------------------------------|--------------------------------------------|---------------------------------------------|-------------------------------------------------------------|
| `git stash` / `git commit`        | –                                          | Saves your WIP (stash or commit)            | Tracks/stashes changes; cleans tracked files                |
| `git fetch origin`                | Download new objects & refs from origin    | Updates `origin/*` remote-tracking branches | No change                                                   |
| `git branch contour origin/contour` | –                                        | Creates local `contour` branch ref          | No change                                                   |
| `git checkout <branch>`           | –                                          | Moves HEAD; resets index to that branch’s tree | Overwrites/creates/deletes **tracked** files; leaves **untracked** files alone |
| `git pull --ff-only origin master`| Fetch remote (if needed)                  | Fast-forwards `master` to `origin/master`   | Updates tracked files to new tip                            |
| `git merge --no-ff contour`       | –                                          | Creates merge commit on `master`            | Applies merged content; conflict markers if needed          |
| `git add` + `git commit`          | –                                          | Finalizes merge commit                      | Cleans conflict markers                                     |
| `git push origin master`          | Updates remote `master`                    | –                                           | No change                                                   |



# Revert to an Earlier Commit

Follow these steps to roll your `master` branch back to a known commit (e.g. `d48da7dfee02edac332bcd7101948f4db40f812e`), verify it, and force-push the change to `origin`.

---

1. **Find the commit ID**

   ```bash
   git log
   # … locate the hash you want (e.g. d48da7dfee02edac332bcd7101948f4db40f812e)
   ```

2. **Inspect that commit (optional)**

   ```bash
   git checkout d48da7dfee02edac332bcd7101948f4db40f812e
   # ⮕ your WD and index now match that snapshot; open files to confirm
   ```

3. **Return to master**

   ```bash
   git checkout master
   git branch
   # ⮕ switch back to your master branch, check where the HEAD is
   ```

4. **Reset local master to the chosen commit**

   ```bash
   git reset --hard d48da7dfee02edac332bcd7101948f4db40f812e
   # ⮕ master’s tip, index, and WD now match that commit exactly
   ```

5. **Force-push the reset to GitHub**

   ```bash
   git push origin master --force-with-lease
   # ⮕ origin/master is rewritten to match your local master
   ```

6. **(Optional) Sync down again**

   ```bash
   git pull origin master
   # ⮕ confirm your local WD/index match the newly updated origin
   ```

---

> **Notes:**
> - `--hard` will discard any uncommitted work—stash or commit first if needed.
> - Prefer `--force-with-lease` over `--force` to avoid clobbering remote changes you haven’t seen.
> - Once reset and pushed, the orphaned commits are no longer on `master` (but still retrievable by hash until garbage-collected).

