"""Content for the /examples/<slug> SEO landing pages -- Typortrait.com and
PawsInWords.com only (see ops/GALLERY-SOURCES.md for the source-photo prompts
and the category -> image-id assignments this content is built against).

Each entry's `images` list gives the id (matching a file pair under
static/examples/<slug>/<id>-before.jpg + <id>-after.png) and a short label
used for that pair's alt text and caption. A category renders with whichever
images are actually present on disk -- so this can ship before every photo
exists, and fills in as files are added with no code change.
"""

EXAMPLES = {
    "portraits-for-parents-grandparents": {
        "brand": "typortrait",
        "eyebrow": "Gift Guide",
        "h1": "Portrait Gifts for Parents & Grandparents",
        "title": "Portrait Gifts for Parents & Grandparents — Typortrait",
        "meta": ("A meaningful gift for Mom, Dad, or grandparents: a portrait built "
                 "from their photo and the words that describe them — more lasting "
                 "than another card."),
        "intro_html": (
            "<p>Past a certain age, most parents and grandparents will tell you "
            "they don't need another thing. What they'll actually hang on the "
            "wall is different: a portrait that's unmistakably <i>them</i>, made "
            "from a photo you already love and the words that describe who they "
            "are — their name, the grandkids' names, the qualities everyone "
            "already says about them.</p>"
            "<p>It works from any clear photo — birthdays, Mother's and Father's "
            "Day, a milestone anniversary, or no occasion at all.</p>"),
        "faq": [
            ("What photo works best?",
             "A clear, well-lit photo where the face is facing mostly toward the "
             "camera and the eyes are open. Studio portraits, phone photos, even "
             "older scanned prints all work."),
            ("Can I use their name and other specific words?",
             "Yes — you choose exactly which words appear. Most people use a "
             "name, a few qualities, and the names of children or grandchildren."),
            ("How long does it take?",
             "The preview renders in under a minute so you can see it before you "
             "buy. Prints ship separately once you order."),
        ],
        "images": [
            {"id": "H04", "label": "Grandmother portrait, word art gift"},
            {"id": "H05", "label": "Grandfather portrait, golden-hour word art"},
            {"id": "H15", "label": "Mother portrait, word art keepsake"},
            {"id": "H16", "label": "Father portrait, word art gift"},
        ],
    },
    "couple-anniversary-portraits": {
        "brand": "typortrait",
        "eyebrow": "Gift Guide",
        "h1": "Couple & Anniversary Portraits",
        "title": "Couple & Anniversary Portraits — Typortrait",
        "meta": ("An anniversary or wedding gift that's actually about the two of "
                 "you: a portrait built from your own photo and the words of your "
                 "story."),
        "intro_html": (
            "<p>A first anniversary, a fiftieth, an engagement, a wedding gift for "
            "someone else's — the occasion changes, the idea doesn't: a portrait "
            "of the two of you, made from a photo you already love and the words "
            "that are actually yours. Your names, your wedding date, an inside "
            "joke, the vow line that mattered.</p>"
            "<p>Unlike a generic \"anniversary gift,\" nothing about it is "
            "interchangeable with anyone else's.</p>"),
        "faq": [
            ("Does it work with a photo of two people?",
             "Yes — the engine handles couples and small groups; each face is "
             "detected and rendered individually within one portrait."),
            ("Can we use our wedding date or vows as the words?",
             "Yes, any short words or phrases you choose — names, a date, a line "
             "from your vows."),
            ("What if we want it printed for a gift?",
             "Digital download is available immediately; framed and fine-art "
             "prints ship separately."),
        ],
        "images": [
            {"id": "H10", "label": "Couple portrait, word art gift"},
            {"id": "H19", "label": "Couple portrait at golden hour, word art"},
            {"id": "H20", "label": "Senior couple anniversary portrait, word art"},
        ],
    },
    "family-portraits": {
        "brand": "typortrait",
        "eyebrow": "Gift Guide",
        "h1": "Family Portraits",
        "title": "Family Portraits, Woven From Your Own Words — Typortrait",
        "meta": ("A family portrait built from your own photo and the names and "
                 "words that belong to your family — for the wall, not the drawer."),
        "intro_html": (
            "<p>Most family photos end up on a phone and nowhere else. This turns "
            "one into something meant to be looked at — a portrait built from "
            "the photo itself, made of your family's own names, a shared "
            "nickname, the words that describe your household.</p>"
            "<p>Works for two people or a full multi-generation group in one "
            "frame.</p>"),
        "faq": [
            ("How many people can be in one portrait?",
             "The engine detects every face in the photo, from a couple to a "
             "full family group."),
            ("Can grandparents, parents, and kids all be in the same portrait?",
             "Yes — multi-generation group photos work the same as any other."),
            ("What words should we use?",
             "Most families use their last name or a household nickname plus a "
             "few words that describe them — but it's entirely your choice."),
        ],
        "images": [
            {"id": "H11", "label": "Family portrait, word art"},
            {"id": "H18", "label": "Friends group portrait, word art"},
            {"id": "H21", "label": "Family of four portrait, word art"},
            {"id": "H22", "label": "Three-generation family portrait, word art"},
        ],
    },
    "kids-baby-portraits": {
        "brand": "typortrait",
        "eyebrow": "Nursery & Keepsake",
        "h1": "Kids & Baby Portraits",
        "title": "Kids & Baby Portraits for the Nursery — Typortrait",
        "meta": ("A baby or child's portrait built from your own photo and their "
                 "name — nursery art or a keepsake that grows with them."),
        "intro_html": (
            "<p>A baby photo changes fast; a portrait doesn't have to. This "
            "builds one from a favorite photo and your child's name, birth date, "
            "or a few words you'd want them to grow up hearing — nursery wall "
            "art now, a keepsake later.</p>"
            "<p>Works from a newborn photo through early childhood — any age, "
            "any clear shot.</p>"),
        "faq": [
            ("What age range works best?",
             "Any age from newborn through childhood — what matters is a clear, "
             "well-lit photo with the face visible."),
            ("Can we include the birth date or birth weight?",
             "Yes — any words you choose, including a date, weight, or a short "
             "welcome message."),
            ("Is it safe to use a photo of my child?",
             "Photos are processed to build the portrait and are not kept beyond "
             "what's needed to deliver your order — see our privacy policy for "
             "specifics."),
        ],
        "images": [
            {"id": "H06", "label": "Child portrait, laughing, word art"},
            {"id": "H23", "label": "Baby portrait, nursery word art"},
            {"id": "H24", "label": "Siblings portrait, word art"},
            {"id": "H25", "label": "Toddler portrait, word art"},
        ],
    },
    "dog-portraits": {
        "brand": "pawsinwords",
        "eyebrow": "Gift Guide",
        "h1": "Dog Portraits, Made From Their Photo",
        "title": "Custom Dog Portraits, Word Art — Paws in Words",
        "meta": ("A custom dog portrait built from your own photo and the words "
                 "that describe them — their name, their quirks, their whole "
                 "personality in one piece."),
        "intro_html": (
            "<p>Anyone can print a photo of their dog. This builds a portrait "
            "out of one instead — their name, the nicknames, the words every "
            "dog owner already uses (loyal, goofy, velcro dog, good boy) woven "
            "into a likeness that's unmistakably <i>your</i> dog.</p>"
            "<p>Works for any breed, coat color, and coat length — from a "
            "close-up face shot to a full-body action photo.</p>"),
        "faq": [
            ("What photo works best?",
             "A clear, well-lit photo with the dog's face visible and eyes "
             "open — close-up or full-body both work."),
            ("Does it work for any breed or coat color?",
             "Yes, including dark coats, light coats, curly, and long-haired "
             "breeds."),
            ("Can I use my dog's name as the words?",
             "Yes — most people use their dog's name plus a few words that "
             "describe them."),
        ],
        "images": [
            {"id": "P01", "label": "Golden Retriever word art portrait"},
            {"id": "P02", "label": "Black Labrador word art portrait"},
            {"id": "P03", "label": "Poodle word art portrait"},
            {"id": "P04", "label": "Border Collie word art portrait"},
            {"id": "P08", "label": "Fluffy white dog word art portrait"},
            {"id": "P11", "label": "Corgi word art portrait"},
            {"id": "P13", "label": "German Shepherd word art portrait"},
        ],
    },
    "cat-portraits": {
        "brand": "pawsinwords",
        "eyebrow": "Gift Guide",
        "h1": "Cat Portraits, Made From Their Photo",
        "title": "Custom Cat Portraits, Word Art — Paws in Words",
        "meta": ("A custom cat portrait built from your own photo and the words "
                 "that describe them — a likeness that's unmistakably your cat."),
        "intro_html": (
            "<p>Cats don't sit still for much, but a photo is all this needs. "
            "The portrait is built from that photo and the words that are "
            "actually theirs — a name, a personality trait, whatever their "
            "household already calls them.</p>"
            "<p>Works for any coat color, including solid black and dark fur — "
            "the engine is built to keep a real, visible eye rather than a flat "
            "dark shape.</p>"),
        "faq": [
            ("Does it work with a black cat?",
             "Yes — dark coats are handled specifically so the eyes stay "
             "visible rather than reading as a flat shadow."),
            ("What if my cat's eyes are closed in the photo?",
             "You'll see a note before you buy if the photo's eyes are shut, "
             "so you can pick a different one if needed — that way the result "
             "always looks right."),
            ("Can I use my cat's name as the words?",
             "Yes — any words you choose, most often a name plus a trait or two."),
        ],
        "images": [
            {"id": "P05", "label": "Orange tabby cat word art portrait"},
            {"id": "P06", "label": "Black cat word art portrait"},
            {"id": "P07", "label": "Gray cat word art portrait"},
            {"id": "P12", "label": "Calico cat word art portrait"},
        ],
    },
    "pet-memorial-portraits": {
        "brand": "pawsinwords",
        "eyebrow": "In Loving Memory",
        "h1": "Pet Memorial Portraits",
        "title": "Pet Memorial Portraits, Made From Their Photo — Paws in Words",
        "meta": ("A gentle way to remember a pet: a portrait built from a favorite "
                 "photo and the words that will always describe them."),
        "intro_html": (
            "<p>Losing a pet doesn't come with the same rituals people get, but "
            "the loss is no smaller. This is a quiet way to keep them close — a "
            "portrait built from a favorite photo and the words that will "
            "always be theirs: their name, the years, the qualities you'll "
            "always remember.</p>"
            "<p>Take your time choosing the photo. There's no rush, and the "
            "preview is free to look at before you decide on anything.</p>"),
        "faq": [
            ("What photo should I choose?",
             "Whichever one feels most like them — a clear, well-lit photo "
             "with their face visible works best, at any age."),
            ("Can I include their name and the years?",
             "Yes — most people use a name and a short phrase or the years "
             "they were loved."),
            ("Is there a rush to decide?",
             "No — the preview is free to see, and there's no obligation until "
             "you choose to order."),
        ],
        "images": [
            {"id": "P09", "label": "Senior dog memorial word art portrait"},
            {"id": "P15", "label": "Senior cat memorial word art portrait"},
            {"id": "P16", "label": "Dog memorial word art portrait, golden light"},
            {"id": "P17", "label": "Cat memorial word art portrait, resting"},
        ],
    },
}
