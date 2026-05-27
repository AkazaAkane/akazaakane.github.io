# Project Structure

AstroPaper v5.5.1 blog — bilingual (zh/en) static site deployed to GitHub Pages.

## Key Directories

```
src/
  config.ts              # SITE constants
  content.config.ts      # blog collection schema (glob loader from src/data/blog/)
  constants.ts           # SOCIALS, SHARE_LINKS

  data/blog/
    zh/                  # Chinese blog posts, organized by year/month
    en/                  # English blog posts — same filename as zh counterpart to pair translations

  i18n/
    ui.ts                # Translation dictionary for all UI strings; LOCALES, DEFAULT_LOCALE, useTranslations()
    utils.ts             # getLangFromPostId, getBaseSlug, getPostsByLang, getTranslation, getOtherLocale

  pages/
    index.astro          # Redirects / → /en/
    collect.md           # Static collection page (not locale-scoped)
    404.astro
    robots.txt.ts
    rss.xml.ts           # Combined RSS feed for all languages
    og.png.ts

    [lang]/              # Dynamic locale prefix: "zh" | "en"
      index.astro        # Homepage per locale
      search.astro
      archives/index.astro
      posts/
        [...page].astro  # Paginated post list
        [...slug]/
          index.astro    # Individual post
          index.png.ts   # Dynamic OG image

      tags/
        index.astro
        [tag]/[...page].astro

    posts/[...slug]/index.astro   # Backward-compat meta-refresh redirect → /en/posts/...
    tags/[tag]/[...page].astro    # Backward-compat meta-refresh redirect → /en/tags/...

  layouts/
    Layout.astro         # HTML shell; accepts lang prop for <html lang>
    PostDetails.astro    # Post layout; accepts lang, allPosts for translation link
    Main.astro
    AboutLayout.astro

  components/
    Header.astro         # Nav with locale-prefixed links; accepts lang prop
    LanguageSwitcher.astro  # zh ↔ en toggle button shown in nav
    Footer.astro         # Accepts lang prop for translated text
    BackButton.astro     # Accepts lang prop
    Breadcrumb.astro     # Locale-aware breadcrumb
    Card.astro           # Post card; getPath auto-derives locale from post.id
    Tag.astro            # Accepts lang prop for locale-prefixed tag links
    Pagination.astro
    Socials.astro
    Datetime.astro
    EditPost.astro
    ShareLinks.astro
    BackToTopButton.astro

  utils/
    getPath.ts           # Returns /{lang}/posts/{slug}; strips locale + numeric (year/month) dirs from path
    getSortedPosts.ts
    getUniqueTags.ts
    getPostsByTag.ts
    getPostsByGroupCondition.ts
    postFilter.ts
    slugify.ts
    generateOgImages.ts
    transformers/

  styles/
  scripts/
    theme.ts

scripts/
  new-post.mjs           # CLI: `npm run new-post` — scaffolds a new blog post

public/
  img/                   # Static images referenced in posts
  pagefind/              # Pagefind search index (generated on build)

astro.config.ts          # i18n: defaultLocale "en", locales ["zh","en"], prefixDefaultLocale true
```

## Routing

| URL | Source |
|-----|--------|
| `/` | Redirect → `/en/` |
| `/zh/` or `/en/` | `[lang]/index.astro` |
| `/zh/posts/` | `[lang]/posts/[...page].astro` |
| `/zh/posts/{slug}/` | `[lang]/posts/[...slug]/index.astro` |
| `/zh/tags/` | `[lang]/tags/index.astro` |
| `/zh/tags/{tag}/` | `[lang]/tags/[tag]/[...page].astro` |
| `/zh/archives/` | `[lang]/archives/index.astro` |
| `/zh/search/` | `[lang]/search.astro` |
| `/posts/{slug}/` | Meta-refresh → `/en/posts/{slug}/` |
| `/tags/{tag}/` | Meta-refresh → `/en/tags/{tag}/` |
| `/archives/`, `/tags/`, `/search/` | astro.config.ts redirect → `/en/...` |

## Adding a New Blog Post

**Quick way** — run the interactive CLI:

```bash
npm run new-post
# or with flags:
npm run new-post -- --title "My Post" --lang zh --tags "tech,ai" --description "About AI"
```

This creates `src/data/blog/{lang}/{YYYY}/{MM}/{slug}.md` with frontmatter pre-filled.

**Manual way:**

1. Create the file at `src/data/blog/zh/2025/06/my-slug.md` (Chinese)
2. Optionally create `src/data/blog/en/2025/06/my-slug.md` (same filename = paired translation)
3. Year/month folders are for local organization only — URLs stay `/zh/posts/my-slug/`
4. The language switcher in the header automatically links between translations when both exist
