# GitHub Copilot CLI Setup

## Prerequisites

You need:

- An active GitHub Copilot subscription
- PowerShell 6+ on Windows
- Node.js 22+ for the npm installation method

## Install on Windows

Download and install Node.js 22 or later from [nodejs.org](https://nodejs.org/en/download).

Then install Copilot CLI with npm:

```powershell
npm install -g @github/copilot
```

## Install on Linux

Using the official installer:

```bash
curl -fsSL https://gh.io/copilot-install | bash
```

Alternatively, install with npm:

```bash
npm install -g @github/copilot
```

Start Copilot CLI:

```bash
copilot
```

On the first launch, use `/login` and follow the authentication instructions.

## Claude Code

Claude Code requires a supported Claude account, such as Pro, Max, Team, Enterprise, or Console.

### Install on Windows

Run this command in PowerShell:

```powershell
irm https://claude.ai/install.ps1 | iex
```

Alternatively, install it with WinGet:

```powershell
winget install Anthropic.ClaudeCode
```

Git for Windows is optional, but it enables Claude Code's Bash tool.

### Install on Linux

Run the official installer:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Alternatively, install it with npm. Node.js 22+ is required:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify the installation:

```bash
claude --version
```

Start Claude Code from your project directory:

```bash
claude
```

On the first launch, follow the browser prompts to authenticate.

## Install Agent Skills

Skills are reusable instructions stored in a `SKILL.md` file.

### Install a skill with `npx skills`

Search for a skill:

```bash
npx skills find TOPIC
```

List the skills in a repository before installing:

```bash
npx skills add OWNER/REPOSITORY --list
```

Install it:

```bash
npx skills add OWNER/REPOSITORY --skill SKILL --agent github-copilot
```

For example:

```bash
npx skills add github/awesome-copilot --skill documentation-writer --agent github-copilot
```

Use `-g` to install the skill globally for your user account. Without `-g`, it is installed for the current project:

```bash
npx skills add OWNER/REPOSITORY --skill SKILL --agent github-copilot -g
```

List installed skills:

```bash
npx skills list
```

### Add a skill manually

Project skills go in:

```text
.github/skills/<skill-name>/SKILL.md
```

Personal skills shared across projects go in:

```text
~/.copilot/skills/<skill-name>/SKILL.md
```

After adding a skill during a Copilot session, reload skills:

```text
/skills reload
```

List available skills:

```text
/skills list
```

Use a skill explicitly by referring to it by name:

```text
Use the /documentation-writer skill to improve this README.
```

> Review skills before installing them. Skills may contain scripts or instructions that can execute commands.

## Official Documentation

- [Install GitHub Copilot CLI](https://docs.github.com/en/copilot/github-copilot-in-the-cli/installing-github-copilot-in-the-cli)
- [Install Claude Code](https://code.claude.com/docs/en/setup)
- [Add skills to Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills)

## Some good links to check
- [Markitdown](https://github.com/microsoft/markitdown) to convert pdfs, pptx, and other documents to Markdown. This is useful for quickly generating Markdown content from various file formats. Agents tend to work well with Markdown content.
- [Matt Pocock YouTube Channel](https://www.youtube.com/@mattpocockuk) for tutorials on web development and JavaScript. His videos are helpful for understanding complex programming concepts and best practices.
