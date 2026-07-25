"""Editorial / "publisher" content for FaithInWords (pilot).

Kept OUT of catalog.json so the client gallery payload stays lean — this is
server-rendered on the per-item pages, the intent-guide pages, and the story page.

Voice: Catholic-primary with ecumenical edges. Scripture is quoted in the
Douay-Rheims (public-domain, Catholic) translation. The universal Christ pieces in
this pilot carry the DRV too; the KJV/RSV pairings in catalog.json are overridden
here when a page has an entry below.
"""

# id -> {scripture (D-R), scripture_ref, about (HTML)}
PORTRAIT_CONTENT = {
    "jesus-compassionate-savior": {
        "scripture": "Come to me, all you that labour, and are burdened, and I will refresh you.",
        "scripture_ref": "Matthew 11:28 (D-R)",
        "about": (
            "<p>Woven from the mercies the Gospels never tire of naming &mdash; <i>Savior, Compassion, "
            "Mercy, Grace, Forgiveness, Redeemer</i> &mdash; this portrait gathers the tenderness of "
            "Christ into his own face. It is the Lord who wept at a tomb, who touched the leper no one "
            "else would touch, who told the sinner only to go and sin no more.</p>"
            "<p>&ldquo;Come to me, all you that labour, and are burdened, and I will refresh you&rdquo; "
            "(Matthew 11:28). Read closely, the words are a catalogue of everything he does with a "
            "wounded heart; seen whole, they are the face that looks on you the way he looked on the "
            "crowds &mdash; and was moved with compassion.</p>"),
    },
    "jesus-good-shepherd": {
        "scripture": "I am the good shepherd. The good shepherd giveth his life for his sheep.",
        "scripture_ref": "John 10:11 (D-R)",
        "about": (
            "<p>Built from the names Christ gave himself and the ones his people gave back &mdash; "
            "<i>Shepherd, Guide, Gate, the Lamb who leads the flock</i> &mdash; this is an image every "
            "Christian knows in the marrow.</p>"
            "<p>&ldquo;I am the good shepherd. The good shepherd giveth his life for his sheep&rdquo; "
            "(John 10:11). It is the One who leaves the ninety-nine, who calls his own by name, who lays "
            "down his life and takes it up again. Close, it is a field of words; far, it is the face "
            "that goes looking for you.</p>"),
    },
    "jesus-prince-of-peace": {
        "scripture": ("For a CHILD IS BORN to us, and a son is given to us… and his name shall "
                      "be called… the Prince of Peace."),
        "scripture_ref": "Isaias 9:6 (D-R)",
        "about": (
            "<p>Woven from the stillness the prophets promised &mdash; <i>Peace, Serenity, Rest, "
            "Comfort, Shalom</i> &mdash; this portrait carries the title Isaiah gave the coming Messiah "
            "seven centuries before Bethlehem.</p>"
            "<p>&ldquo;For a CHILD IS BORN to us&hellip; and his name shall be called&hellip; the Prince "
            "of Peace&rdquo; (Isaias 9:6). Not the peace the world gives, but the peace he breathed on "
            "the disciples behind locked doors. Read closely, it is a litany of quiet; seen whole, it is "
            "the face that said, in the storm, &ldquo;Peace, be still.&rdquo;</p>"),
    },
    "jesus-light-of-the-world": {
        "scripture": ("I am the light of the world: he that followeth me, walketh not in darkness, but "
                      "shall have the light of life."),
        "scripture_ref": "John 8:12 (D-R)",
        "about": (
            "<p>Made from the radiance the Gospel of John returns to again and again &mdash; <i>Light, "
            "Dawn, Truth, Glory</i> &mdash; this portrait renders Christ as the light the darkness could "
            "not overcome.</p>"
            "<p>&ldquo;I am the light of the world: he that followeth me, walketh not in darkness, but "
            "shall have the light of life&rdquo; (John 8:12). It is the light of the first day of "
            "creation and the light of Easter morning. Close, the words glow like a window; far, they "
            "gather into the face the Church calls the Light of the nations.</p>"),
    },
    "jesus-king-of-kings": {
        "scripture": "And he hath on his garment, and on his thigh written: KING OF KINGS AND LORD OF LORDS.",
        "scripture_ref": "Apocalypse 19:16 (D-R)",
        "about": (
            "<p>Woven from the language of heaven&rsquo;s throne room &mdash; <i>King, Lord, Majesty, "
            "Glory, Sovereign, Reign</i> &mdash; this portrait crowns Christ with the title the "
            "Apocalypse gives the rider called Faithful and True.</p>"
            "<p>&ldquo;And he hath on his garment, and on his thigh written: KING OF KINGS AND LORD OF "
            "LORDS&rdquo; (Apocalypse 19:16). The King whose crown was first made of thorns; the Lord "
            "whose throne was first a cross. Read closely, it is a coronation in words; seen whole, it "
            "is the face before whom every knee shall bend.</p>"),
    },
}


# slug -> intent/gift landing page. `items` are curated catalog ids (order matters).
GUIDES = {
    "confirmation-gifts": {
        "eyebrow": "Gift Guide",
        "h1": "Confirmation Gifts That Mean Something",
        "title": "Confirmation Gifts That Mean Something — Sacred Word Portraits",
        "meta": ("Meaningful Catholic confirmation gifts: word portraits of patron saints, guardian "
                 "angels and Christ the King — a keepsake that stays on the wall long after the card."),
        "intro": (
            "<p>Confirmation seals a young Catholic with the Holy Spirit, and most take a saint&rsquo;s "
            "name to carry for life. A portrait of that <b>patron saint</b> &mdash; or of the Christ they "
            "now publicly follow &mdash; is a gift that outlasts the envelope, hung where it can be seen "
            "and prayed with for years.</p>"
            "<p>Each portrait below is composed entirely of the words that belong to its subject, and "
            "arrives as a keepsake-quality print or an instant digital download.</p>"),
        "items": ["st-michael-the-archangel", "st-therese-of-lisieux", "st-joseph",
                  "st-francis-of-assisi", "guardian-angel", "jesus-king-of-kings",
                  "jesus-good-shepherd", "st-christopher"],
    },
    "sympathy-christian-gifts": {
        "eyebrow": "Gift Guide",
        "h1": "Christian Sympathy Gifts That Comfort",
        "title": "Christian Sympathy Gifts That Comfort — Sacred Word Portraits",
        "meta": ("Christian sympathy and condolence gifts: the Good Shepherd, the Compassionate Savior, "
                 "Our Lady of Sorrows — word portraits that comfort long after the flowers."),
        "intro": (
            "<p>When words fail, give one that holds them. The <b>Good Shepherd</b> who carries the lost "
            "lamb home, the <b>Compassionate Savior</b>, <b>Our Lady of Sorrows</b> who knows a "
            "mother&rsquo;s grief, the <b>Prince of Peace</b> &mdash; images that meet sorrow where it "
            "is, and stay long after the flowers have gone.</p>"
            "<p>Each is woven entirely from the names and titles that belong to its subject &mdash; a "
            "quiet, lasting comfort for a grieving home.</p>"),
        "items": ["jesus-good-shepherd", "jesus-compassionate-savior", "our-lady-of-sorrows",
                  "jesus-prince-of-peace", "the-blessed-virgin-mary", "st-joseph", "guardian-angel"],
    },
    "catholic-wall-art": {
        "eyebrow": "For the Home That Prays",
        "h1": "Catholic Wall Art, Woven From Words",
        "title": "Catholic Wall Art, Woven From Words — Sacred Word Portraits",
        "meta": ("Catholic wall art for the home and prayer room: Marian portraits, the Sacred Heart of "
                 "devotion, angels and saints — each composed entirely of the words that belong to it."),
        "intro": (
            "<p>A home is a <i>domestic church</i>, and a domestic church needs images. Marian devotion "
            "for the mantel, a guardian angel over a child&rsquo;s bed, a patron saint who keeps watch, "
            "the Christ who presides over the table &mdash; each portrait here is made entirely of the "
            "words that belong to its subject, so a likeness read closely becomes a prayer.</p>"
            "<p>Available as archival prints, framed keepsakes, or an instant digital download with free "
            "matching wallpapers.</p>"),
        "items": ["the-blessed-virgin-mary", "our-lady-of-guadalupe", "jesus-good-shepherd",
                  "jesus-compassionate-savior", "st-michael-the-archangel", "st-joseph",
                  "immaculate-heart-of-mary", "jesus-king-of-kings", "guardian-angel",
                  "st-francis-of-assisi", "st-therese-of-lisieux", "jesus-light-of-the-world"],
    },
}


STORY = {
    "eyebrow": "How the portraits are made",
    "h1": "A face, revealed in the words that belong to it",
    "title": "How our sacred word portraits are made — Faith in Words",
    "meta": ("How Faith in Words creates sacred portraits composed entirely of words — the names, "
             "titles and Scripture of each subject, woven into a faithful likeness."),
    "body": (
        "<p class=\"lead\">Every portrait in the collection is built on a simple idea with a long "
        "history: that a face can be written as much as drawn &mdash; that the truest likeness of a "
        "saint might be made of the very words we use to praise them.</p>"

        "<h2>It begins with a name</h2>"
        "<p>We start from a classical depiction of the subject &mdash; the way the Church&rsquo;s "
        "artists have shown them for centuries &mdash; and then set it aside. In its place we rebuild "
        "the likeness, line by line, out of words that belong to that subject alone: their names and "
        "titles, the qualities the faithful have always ascribed to them, phrases from Scripture and "
        "the tradition. Nothing generic; nothing borrowed from another face.</p>"

        "<h2>Read it closely, and it becomes a litany</h2>"
        "<p>Step near a finished portrait and the likeness dissolves into language &mdash; "
        "<i>Shepherd, Mercy, Guadalupe, Queen of Heaven, full of grace</i>. Step back, and the face "
        "returns. That is the moment people remember: the discovery that they are not looking at an "
        "image <i>of</i> the words, but an image <i>made from</i> them.</p>"

        "<h2>Paired with Scripture</h2>"
        "<p>Each piece is joined to a verse chosen for its subject &mdash; quoted in the Douay-Rheims "
        "for the Marian and saintly portraits, so the words on the wall and the Word they point to "
        "belong together.</p>"

        "<h2>Made to keep</h2>"
        "<p>A finished portrait is available as a high-resolution digital download &mdash; with matching "
        "phone and desktop wallpapers included free &mdash; and as archival fine-art prints and framed "
        "keepsakes, printed and shipped to your door. Made, in every case, to be lived with and prayed "
        "before for a long time.</p>"),
}
