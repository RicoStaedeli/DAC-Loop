import subprocess
import sys



def _run_command(command, capture_output=False):
    """Run a shell command and optionally capture output."""
    result = subprocess.run(
        command, shell=True,
        capture_output=capture_output, text=True
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    if capture_output:
        return result.stdout.strip()

def git_commit_and_push(commit_message):
    # Check current branch
    branch = _run_command("git rev-parse --abbrev-ref HEAD", capture_output=True)
    if branch == "main":
        print("You are on the 'main' branch. --> Aborting commit.")
        return

    # Stage all changes
    _run_command("git add .")
    print(f"CAutomatic git add: {branch}")

    # Commit with the given message
    _run_command(f'git commit -m "Automatic commit: {commit_message}"')
    print(f"Automatic git commit: {commit_message}")

    # Push to the current branch
    _run_command("git push")
    print(f"Automatic git push to origin")