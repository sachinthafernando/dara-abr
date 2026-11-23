# Publishing this repository on GitHub

These are one-off steps for the author. Delete this file after publishing if you like.

1. **Fill in the placeholders.**
   - Replace `<your-github-username>` in `README.md` (badge and clone URL). To find every occurrence:
     `grep -rn "<your-github-username>" .`
   - Uncomment `repository-code` in `CITATION.cff`.
2. **Settle the licence** with the supervisors. Replace `LICENSE` with the agreed text, then update the licence badge
   and the "Licence" section of `README.md`. Common choices for research code are MIT, BSD-3-Clause and Apache-2.0.
   The University of Hertfordshire may have a policy on this.
3. **Create the repository** on GitHub (for example `dara-abr`) without a README, licence or .gitignore, then run:

   ```bash
   cd dara-abr
   git init -b main
   git add .
   git commit -m "DARA v1.0.0: code, experiment definitions and results"
   git remote add origin https://github.com/<your-github-username>/dara-abr.git
   git push -u origin main
   git tag -a v1.0.0 -m "DARA v1.0.0" && git push origin v1.0.0
   ```

4. **Create a release** in the GitHub web interface: Releases → Draft a new release → tag `v1.0.0`. Paste the text of
   `docs/RELEASE_NOTES_v1.0.0.md` as the release notes.
5. **Check CI.** The Actions tab should show the `CI` workflow passing. The CPU smoke test runs on demand: Actions →
   CI → Run workflow → tick "smoke".
6. **Optional: archive with a DOI.** Link the repository to [Zenodo](https://zenodo.org/account/settings/github/)
   *before* creating the release; Zenodo then archives every release and assigns a DOI. Add the DOI badge to the
   README and the `doi` field to `CITATION.cff`.
7. **Optional: model checkpoints.** The trained checkpoints (`runs/train/*/checkpoint.pt`) are not in the
   repository. To let others replay policies without training, upload a `runs.tar.gz` of the checkpoints as a release
   asset or to Zenodo, and link it from the README.
