export const LOCALES = ["zh", "en"] as const;
export type Locale = (typeof LOCALES)[number];
export const DEFAULT_LOCALE: Locale = "en";

export const ui = {
  zh: {
    // Nav
    "nav.posts": "文章",
    "nav.tags": "标签",
    "nav.collect": "收藏",
    "nav.archives": "归档",
    "nav.search": "搜索",
    "nav.skipToContent": "跳至正文",
    "nav.openMenu": "打开菜单",
    "nav.closeMenu": "关闭菜单",

    // Index page
    "index.featured": "精选",
    "index.recentPosts": "最新文章",
    "index.allPosts": "全部文章",
    "index.socialLinks": "社交链接：",

    // Posts page
    "posts.pageTitle": "文章",
    "posts.pageDesc": "所有已发布的文章。",

    // Tags page
    "tags.pageTitle": "标签",
    "tags.pageDesc": "所有文章中使用的标签。",
    "tags.tagLabel": "标签：",
    "tags.tagPageDesc": '所有带有标签"{tag}"的文章。',

    // Archives page
    "archives.pageTitle": "归档",
    "archives.pageDesc": "所有已归档的文章。",
    "archives.jan": "一月",
    "archives.feb": "二月",
    "archives.mar": "三月",
    "archives.apr": "四月",
    "archives.may": "五月",
    "archives.jun": "六月",
    "archives.jul": "七月",
    "archives.aug": "八月",
    "archives.sep": "九月",
    "archives.oct": "十月",
    "archives.nov": "十一月",
    "archives.dec": "十二月",

    // Search page
    "search.pageTitle": "搜索",
    "search.pageDesc": "搜索任意文章…",

    // Post detail
    "post.prevPost": "上一篇",
    "post.nextPost": "下一篇",
    "post.readInLang": "阅读英文版",

    // Breadcrumb
    "breadcrumb.home": "首页",
    "breadcrumb.postsPage": "文章（第{n}页）",

    // Footer
    "footer.allRightsReserved": "保留所有权利。",

    // Back button
    "back.goBack": "返回",

    // Language switcher
    "lang.switch": "EN",
    "lang.switchLabel": "Switch to English",
    "lang.notAvailable": "暂无英文版",
  },
  en: {
    // Nav
    "nav.posts": "Posts",
    "nav.tags": "Tags",
    "nav.collect": "Collect",
    "nav.archives": "Archives",
    "nav.search": "Search",
    "nav.skipToContent": "Skip to content",
    "nav.openMenu": "Open Menu",
    "nav.closeMenu": "Close Menu",

    // Index page
    "index.featured": "Featured",
    "index.recentPosts": "Recent Posts",
    "index.allPosts": "All Posts",
    "index.socialLinks": "Social Links:",

    // Posts page
    "posts.pageTitle": "Posts",
    "posts.pageDesc": "All the articles I've posted.",

    // Tags page
    "tags.pageTitle": "Tags",
    "tags.pageDesc": "All the tags used in posts.",
    "tags.tagLabel": "Tag:",
    "tags.tagPageDesc": 'All the articles with the tag "{tag}".',

    // Archives page
    "archives.pageTitle": "Archives",
    "archives.pageDesc": "All the articles I've archived.",
    "archives.jan": "January",
    "archives.feb": "February",
    "archives.mar": "March",
    "archives.apr": "April",
    "archives.may": "May",
    "archives.jun": "June",
    "archives.jul": "July",
    "archives.aug": "August",
    "archives.sep": "September",
    "archives.oct": "October",
    "archives.nov": "November",
    "archives.dec": "December",

    // Search page
    "search.pageTitle": "Search",
    "search.pageDesc": "Search any article ...",

    // Post detail
    "post.prevPost": "Previous Post",
    "post.nextPost": "Next Post",
    "post.readInLang": "Read in Chinese",

    // Breadcrumb
    "breadcrumb.home": "Home",
    "breadcrumb.postsPage": "Posts (page {n})",

    // Footer
    "footer.allRightsReserved": "All rights reserved.",

    // Back button
    "back.goBack": "Go back",

    // Language switcher
    "lang.switch": "中文",
    "lang.switchLabel": "切换至中文",
    "lang.notAvailable": "No Chinese version",
  },
} as const;

export type TranslationKey = keyof (typeof ui)["en"];

export function useTranslations(lang: Locale) {
  return function t(key: TranslationKey): string {
    return (ui[lang] as Record<string, string>)[key] ?? (ui["en"] as Record<string, string>)[key] ?? key;
  };
}
