# Vel Documentation

This directory contains the complete documentation for Vel, a 12-Factor Agent Runtime.

**📚 View the live documentation at:** https://rscheiwe.github.io/vel

## Theme

The documentation uses the [Just the Docs](https://just-the-docs.github.io/just-the-docs/) Jekyll theme, which provides:
- 🔍 Built-in search
- 📱 Mobile responsive design
- 🌙 Dark mode
- 📋 Copy code button
- 🔗 Anchor links for all headings

## Documentation Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── index.md                 # Homepage
├── getting-started.md       # Installation and quick start
├── sessions.md              # Multi-turn conversation management
├── prompts.md               # Prompt template system
├── providers.md             # LLM provider configuration
├── tools.md                 # Custom tool creation
├── stream-protocol.md       # Stream event reference
├── api-reference.md         # Complete API documentation
└── 12-factor-alignment.md   # 12-Factor Agent principles
```

## Local Development

### Prerequisites

- Ruby 3.0 or higher
- Bundler

### Setup

```bash
# Navigate to docs directory
cd docs

# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Visit http://localhost:4000/vel
```

The site will automatically rebuild when you edit files.

### Testing Changes

Before committing, test your changes locally:

```bash
# Check for broken links
bundle exec jekyll build

# Serve and test in browser
bundle exec jekyll serve --livereload
```

## Deployment

Documentation automatically deploys to GitHub Pages when you push to the `main` branch:

1. **GitHub Actions** builds the Jekyll site
2. **GitHub Pages** hosts the built site at https://rscheiwe.github.io/vel

No manual deployment needed!

## Contributing

To improve documentation:

1. Edit the relevant `.md` file
2. Add front matter if creating a new page:
   ```yaml
   ---
   layout: default
   title: Page Title
   nav_order: 10
   ---
   ```
3. Test locally (see above)
4. Submit a pull request
5. Changes deploy automatically on merge

## Customization

### Changing Theme Colors

Edit `_config.yml`:

```yaml
color_scheme: dark  # or "light", "nil" (default)
```

### Adding Navigation

Pages automatically appear in navigation based on `nav_order` in front matter:

```yaml
---
title: New Page
nav_order: 5  # Position in sidebar
---
```

### Custom CSS

Create `assets/css/custom.scss`:

```scss
---
---
@import "{{ site.theme }}";

// Your custom styles here
.custom-class {
  color: #ff0000;
}
```

## Resources

- [Just the Docs Documentation](https://just-the-docs.github.io/just-the-docs/)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
