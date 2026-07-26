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

    # --- The Blessed Mother -------------------------------------------------
    "the-blessed-virgin-mary": {
        "scripture": "Hail, full of grace, the Lord is with thee: blessed art thou among women.",
        "scripture_ref": "Luke 1:28 (D-R)",
        "about": (
            "<p>Woven from the names the faithful have called her for two thousand years &mdash; "
            "<i>Mother, Virgin, full of grace, Queen of Heaven, Intercessor</i> &mdash; this portrait "
            "gathers Mary&rsquo;s titles into her likeness. Each is a prayer the Church has never stopped "
            "praying.</p>"
            "<p>&ldquo;Hail, full of grace, the Lord is with thee: blessed art thou among women&rdquo; "
            "(Luke 1:28). From the Annunciation to the foot of the Cross she is the one who said <i>yes</i> "
            "and never unsaid it. Read closely, it is a litany; seen whole, it is the Mother who points "
            "always to her Son.</p>"),
    },
    "immaculate-heart-of-mary": {
        "scripture": "But Mary kept all these words, pondering them in her heart.",
        "scripture_ref": "Luke 2:19 (D-R)",
        "about": (
            "<p>Made from the movements of a mother&rsquo;s heart &mdash; <i>Love, Purity, Sorrow, Joy, "
            "Compassion, Peace</i> &mdash; this portrait renders the Immaculate Heart the Church holds up "
            "beside the Sacred Heart of her Son.</p>"
            "<p>&ldquo;But Mary kept all these words, pondering them in her heart&rdquo; (Luke 2:19). Hers "
            "is the heart that treasured every word about Jesus, that a sword would one day pierce, and "
            "that the faithful entrust their own hearts to still. Close, it is devotion written out; far, "
            "it is the face of a love wholly given.</p>"),
    },
    "our-lady-of-sorrows": {
        "scripture": "And thy own soul a sword shall pierce.",
        "scripture_ref": "Luke 2:35 (D-R)",
        "about": (
            "<p>Woven from the seven sorrows the Church has counted for centuries &mdash; <i>the prophecy "
            "of Simeon, the flight into Egypt, the loss in the Temple, the meeting on the road to "
            "Calvary, the Cross</i> &mdash; this portrait honours the Mother who stood by her Son to the "
            "end.</p>"
            "<p>Simeon foretold it in the Temple: &ldquo;And thy own soul a sword shall pierce&rdquo; "
            "(Luke 2:35). She is <i>Mater Dolorosa</i>, the one who knows grief from the inside &mdash; "
            "which is why the sorrowing have always turned to her. Close, it is a mother&rsquo;s suffering "
            "written out; far, it is compassion that has been through the fire.</p>"),
    },
    "our-lady-of-guadalupe": {
        "scripture": ("A woman clothed with the sun, and the moon under her feet, and on her head a crown "
                      "of twelve stars."),
        "scripture_ref": "Apocalypse 12:1 (D-R)",
        "about": (
            "<p>Woven from the tradition of Tepeyac &mdash; <i>Guadalupe, Juan Diego, the roses, the "
            "tilma, Mother of Mercy</i> &mdash; this portrait honours the Virgin who, as the tradition "
            "holds, appeared in 1531 as a <i>mestiza</i> to a poor convert and left her image on his "
            "cloak.</p>"
            "<p>The Church reads her in the sign of the Apocalypse: &ldquo;A woman clothed with the sun, "
            "and the moon under her feet, and on her head a crown of twelve stars&rdquo; (Apocalypse "
            "12:1). Patroness of the Americas, she is the Mother who came to the poorest and asked, "
            "<i>Am I not here, I who am your Mother?</i></p>"),
    },
    "mary-queen-of-heaven": {
        "scripture": "The queen stood on thy right hand, in gilded clothing; surrounded with variety.",
        "scripture_ref": "Psalm 44:10 (D-R)",
        "about": (
            "<p>Made from the glory the Church ascribes to her &mdash; <i>Queen of Heaven, Immaculate "
            "Conception, Assumption, perpetual Virgin</i> &mdash; this portrait crowns the Mother taken "
            "up body and soul and seated beside her Son.</p>"
            "<p>The psalmist sang of her long before: &ldquo;The queen stood on thy right hand, in gilded "
            "clothing&rdquo; (Psalm 44:10). She reigns not by her own power but as the first of the "
            "redeemed, the creature raised highest by grace. Close, it is a coronation in words; far, it "
            "is the face the litanies hail as <i>Regina caeli</i>.</p>"),
    },

    # --- Angels -------------------------------------------------------------
    "st-michael-the-archangel": {
        "scripture": "And there was a great battle in heaven, Michael and his angels fought with the dragon.",
        "scripture_ref": "Apocalypse 12:7 (D-R)",
        "about": (
            "<p>Built from the titles of heaven&rsquo;s captain &mdash; <i>Archangel, Defender, "
            "Vanquisher of evil, Protector</i> &mdash; this portrait shows the prince of the heavenly "
            "host, sword in hand, the one whose name means <i>Who is like God?</i></p>"
            "<p>&ldquo;And there was a great battle in heaven, Michael and his angels fought with the "
            "dragon&rdquo; (Apocalypse 12:7). The Church has prayed to him against evil for centuries "
            "&mdash; <i>Saint Michael the Archangel, defend us in battle</i>. Close, it is spiritual "
            "warfare in words; far, it is the face that stands between the faithful and the enemy.</p>"),
    },
    "st-raphael-the-archangel": {
        "scripture": "I am the angel Raphael, one of the seven, who stand before the Lord.",
        "scripture_ref": "Tobias 12:15 (D-R)",
        "about": (
            "<p>Woven from a mission of healing and companionship &mdash; <i>Healer, Divine messenger, "
            "Companion</i> &mdash; this portrait honours the archangel whose name means <i>God heals</i>, "
            "the guide of the Book of Tobit.</p>"
            "<p>&ldquo;I am the angel Raphael, one of the seven, who stand before the Lord&rdquo; (Tobias "
            "12:15). He walked the long road with young Tobias, restored a father&rsquo;s sight, and is "
            "invoked still by travellers, the sick, and those seeking a holy spouse. Read closely, it is "
            "healing named; seen whole, it is the face of a friend sent by God.</p>"),
    },
    "guardian-angel": {
        "scripture": "For he hath given his angels charge over thee; to keep thee in all thy ways.",
        "scripture_ref": "Psalm 90:11 (D-R)",
        "about": (
            "<p>Made from the quiet work of heaven &mdash; <i>Protector, Guide, Watchful care, Holy "
            "presence</i> &mdash; this portrait honours the angel God gives each soul to walk beside it "
            "from birth.</p>"
            "<p>&ldquo;For he hath given his angels charge over thee; to keep thee in all thy ways&rdquo; "
            "(Psalm 90:11). Our Lord said their angels always behold the face of the Father. Close, it is "
            "watchfulness in words; far, it is the companion who has never once left your side.</p>"),
    },

    # --- Apostles & Prophets ------------------------------------------------
    "st-peter": {
        "scripture": "Thou art Peter; and upon this rock I will build my church.",
        "scripture_ref": "Matthew 16:18 (D-R)",
        "about": (
            "<p>Built from the name Christ gave a fisherman &mdash; <i>Rock, Apostle, Keys of Heaven, "
            "Martyr</i> &mdash; this portrait honours Simon whom the Lord renamed Peter, first among the "
            "Twelve.</p>"
            "<p>&ldquo;Thou art Peter; and upon this rock I will build my church&rdquo; (Matthew 16:18). "
            "He walked on water and sank, denied his Lord and wept, and was handed the keys all the same "
            "&mdash; then died on a cross in Rome, upside down. Close, it is a commission in words; far, "
            "it is the face of the Church&rsquo;s first shepherd.</p>"),
    },
    "st-paul": {
        "scripture": "I have fought a good fight, I have finished my course, I have kept the faith.",
        "scripture_ref": "2 Timothy 4:7 (D-R)",
        "about": (
            "<p>Woven from the road that turned him around &mdash; <i>Apostle, Damascus, Letters to the "
            "churches, Conversion</i> &mdash; this portrait honours the persecutor struck blind and made "
            "the Church&rsquo;s greatest missionary.</p>"
            "<p>At the end he could write: &ldquo;I have fought a good fight, I have finished my course, I "
            "have kept the faith&rdquo; (2 Timothy 4:7). From a light on the Damascus road to letters "
            "that still shape the faith, no one carried the Gospel farther. Read closely, it is a "
            "life&rsquo;s labour in words; seen whole, it is the face of grace that overtakes a man.</p>"),
    },
    "st-john-the-baptist": {
        "scripture": "Behold the Lamb of God, behold him who taketh away the sin of the world.",
        "scripture_ref": "John 1:29 (D-R)",
        "about": (
            "<p>Made from his one great task &mdash; <i>Prophet, Wilderness preacher, Herald of the "
            "Messiah, Baptism, Repentance</i> &mdash; this portrait honours the voice crying in the "
            "desert, the last of the prophets and the forerunner of Christ.</p>"
            "<p>Seeing Jesus approach, he said: &ldquo;Behold the Lamb of God, behold him who taketh away "
            "the sin of the world&rdquo; (John 1:29). He baptised the Lord in the Jordan and then stepped "
            "aside &mdash; <i>he must increase, but I must decrease</i>. Close, it is a herald&rsquo;s cry "
            "in words; far, it is the face that pointed away from itself.</p>"),
    },
    "moses": {
        "scripture": "And there arose no more a prophet in Israel like unto Moses, whom the Lord knew face to face.",
        "scripture_ref": "Deuteronomy 34:10 (D-R)",
        "about": (
            "<p>Woven from the great deliverance &mdash; <i>Law-giver, Exodus, Ten Commandments, Red Sea, "
            "the Burning Bush, Sinai, Covenant</i> &mdash; this portrait honours the prophet who led "
            "Israel out of slavery and came down the mountain with the Law.</p>"
            "<p>Scripture&rsquo;s own epitaph: &ldquo;And there arose no more a prophet in Israel like "
            "unto Moses, whom the Lord knew face to face&rdquo; (Deuteronomy 34:10). He met God in fire, "
            "parted the sea, and carried a people forty years toward a land he would see but not enter. "
            "Read closely, it is a covenant in words; seen whole, it is the face that spoke with God as a "
            "man speaks to his friend.</p>"),
    },
    "king-david": {
        "scripture": "The Lord ruleth me: and I shall want nothing.",
        "scripture_ref": "Psalm 22:1 (D-R)",
        "about": (
            "<p>Built from a shepherd&rsquo;s rise &mdash; <i>King of Israel, Psalmist, Goliath, "
            "Bethlehem, the harp, Repentance, Covenant</i> &mdash; this portrait honours the boy who "
            "felled a giant and the king from whose line the Messiah would come.</p>"
            "<p>His most beloved psalm begins: &ldquo;The Lord ruleth me: and I shall want nothing&rdquo; "
            "(Psalm 22:1). Shepherd and sinner, warrior and poet, he danced before the Ark and wept for "
            "his sins &mdash; a man after God&rsquo;s own heart. Close, it is a psalm in words; far, it is "
            "the face of Bethlehem&rsquo;s king.</p>"),
    },
    "st-mary-magdalene": {
        "scripture": "Jesus saith to her: Mary. She turning, saith to him: Rabboni (which is to say, Master).",
        "scripture_ref": "John 20:16 (D-R)",
        "about": (
            "<p>Woven from her faithfulness to the end and past it &mdash; <i>Disciple, Witness, "
            "Resurrection, the spices, the empty tomb, Penitent, Love</i> &mdash; this portrait honours "
            "the woman who stayed at the Cross and came first to the grave.</p>"
            "<p>In the garden, mistaking him for the gardener, she heard him speak her name: &ldquo;Jesus "
            "saith to her: Mary&rdquo; &mdash; and she turned and said, &ldquo;Rabboni&rdquo; (John "
            "20:16). The first witness of the Resurrection, the <i>Apostle to the Apostles</i>. Close, it "
            "is devotion in words; far, it is the face that love would not let leave the tomb.</p>"),
    },

    # --- Beloved Patron Saints ---------------------------------------------
    "st-joseph": {
        "scripture": "Joseph, son of David, fear not to take unto thee Mary thy wife.",
        "scripture_ref": "Matthew 1:20 (D-R)",
        "about": (
            "<p>Made from a quiet, faithful strength &mdash; <i>Carpenter, Guardian, Holy Family, "
            "Nazareth, Obedience</i> &mdash; this portrait honours the just man God trusted with his Son "
            "and his Mother.</p>"
            "<p>An angel told him in a dream: &ldquo;Joseph, son of David, fear not to take unto thee "
            "Mary thy wife&rdquo; (Matthew 1:20). He never speaks a word in the Gospels; he only obeys "
            "&mdash; and so became patron of the universal Church, of workers, and of a happy death. "
            "Close, it is fidelity in words; far, it is the face of the man who guarded the Word made "
            "flesh.</p>"),
    },
    "st-francis-of-assisi": {
        "scripture": "Go, sell what thou hast, and give to the poor, and thou shalt have treasure in heaven.",
        "scripture_ref": "Matthew 19:21 (D-R)",
        "about": (
            "<p>Woven from a joyful poverty &mdash; <i>Poverty, Humility, the birds, the stigmata, Assisi, "
            "Brother Sun, rebuild my church</i> &mdash; this portrait honours the merchant&rsquo;s son "
            "who gave away everything to follow Christ crucified.</p>"
            "<p>He took the Gospel at its word: &ldquo;Go, sell what thou hast, and give to the "
            "poor&rdquo; (Matthew 19:21). He preached to birds, wrote a canticle to creation, and bore "
            "the wounds of Christ in his own flesh. Close, it is a life stripped bare in words; far, it "
            "is the face of God&rsquo;s little poor man, <i>il Poverello</i>.</p>"),
    },
    "st-anthony-of-padua": {
        "scripture": ("What woman having ten groats, if she lose one groat, doth not light a candle, and "
                      "sweep the house, and seek diligently until she find it?"),
        "scripture_ref": "Luke 15:8 (D-R)",
        "about": (
            "<p>Made from a preacher&rsquo;s fire and a people&rsquo;s affection &mdash; <i>Miracles, "
            "lost things, Padua, the lily, the Child Jesus</i> &mdash; this portrait honours the "
            "Franciscan whose sermons drew crowds and whose help the faithful still beg for what is "
            "lost.</p>"
            "<p>The Lord asked: &ldquo;What woman having ten groats, if she lose one, doth not light a "
            "candle&hellip; and seek diligently until she find it?&rdquo; (Luke 15:8). A Doctor of the "
            "Church loved most for the smallest of favours &mdash; <i>Tony, Tony, turn around</i>. Close, "
            "it is a search rewarded in words; far, it is the face that holds the Christ Child close.</p>"),
    },
    "st-jude-thaddeus": {
        "scripture": "Keep yourselves in the love of God, waiting for the mercy of our Lord Jesus Christ, unto life everlasting.",
        "scripture_ref": "Jude 1:21 (D-R)",
        "about": (
            "<p>Woven from a stubborn hope &mdash; <i>Apostle, Hope, desperate cases, Miracles, "
            "Martyrdom</i> &mdash; this portrait honours the apostle the faithful call on when every "
            "other door has closed.</p>"
            "<p>His own letter urges: &ldquo;Keep yourselves in the love of God, waiting for the mercy of "
            "our Lord Jesus Christ&rdquo; (Jude 1:21). Long confused with Judas the betrayer, and so left "
            "unasked, he became the patron of hopeless causes &mdash; the saint of last resort who has "
            "never been a last resort in vain. Close, it is hope in words; far, it is the face turned "
            "toward the impossible.</p>"),
    },
    "st-therese-of-lisieux": {
        "scripture": "Unless you be converted, and become as little children, you shall not enter into the kingdom of heaven.",
        "scripture_ref": "Matthew 18:3 (D-R)",
        "about": (
            "<p>Made from a small and radiant holiness &mdash; <i>the Little Way, Simplicity, Carmelite, "
            "flowers, Prayer, Lisieux</i> &mdash; this portrait honours the young nun who found sanctity "
            "not in great deeds but in little ones done with great love.</p>"
            "<p>She built her whole life on one saying: &ldquo;Unless you be converted, and become as "
            "little children, you shall not enter into the kingdom of heaven&rdquo; (Matthew 18:3). Dead "
            "at twenty-four, she promised to spend her heaven doing good on earth, and to let fall a "
            "shower of roses. A Doctor of the Church in the shape of a child. Close, it is the Little Way "
            "in words; far, it is the face of love made simple.</p>"),
    },
    "st-christopher": {
        "scripture": "When thou shalt pass through the waters, I will be with thee, and the rivers shall not cover thee.",
        "scripture_ref": "Isaias 43:2 (D-R)",
        "about": (
            "<p>Woven from an old and beloved legend &mdash; <i>Traveller&rsquo;s patron, Martyr, who "
            "carried the Christ Child, Strength, Protection</i> &mdash; this portrait honours the giant "
            "who, as the tradition tells it, carried travellers across a river and one day bore a child "
            "who grew heavier than the world.</p>"
            "<p>The prophet&rsquo;s promise fits him: &ldquo;When thou shalt pass through the waters, I "
            "will be with thee&rdquo; (Isaias 43:2). His name means <i>Christ-bearer</i>, and the child "
            "he carried was Christ himself. Patron of travellers and drivers, kept close on many a "
            "dashboard. Close, it is protection in words; far, it is the face that carried the weight of "
            "the world across.</p>"),
    },
    "st-rita-of-cascia": {
        "scripture": "With men this is impossible: but with God all things are possible.",
        "scripture_ref": "Matthew 19:26 (D-R)",
        "about": (
            "<p>Made from a life that knew every state &mdash; <i>wife, mother, Augustinian nun, "
            "forgiveness, impossible causes, the thorn wound</i> &mdash; this portrait honours the saint "
            "who forgave her husband&rsquo;s murderers and became patroness of what cannot be done.</p>"
            "<p>Her whole life answers the Lord&rsquo;s word: &ldquo;With men this is impossible: but with "
            "God all things are possible&rdquo; (Matthew 19:26). Wife, widow, and at last the nun she had "
            "longed to be, marked on her brow by a thorn from the Crucified. With Saint Jude she is "
            "invoked for the hopeless. Close, it is impossible causes in words; far, it is the face of "
            "peace won through forgiveness.</p>"),
    },
    "st-joan-of-arc": {
        "scripture": "Take courage, and be strong. Fear not and be not dismayed: because the Lord thy God is with thee.",
        "scripture_ref": "Josue 1:9 (D-R)",
        "about": (
            "<p>Woven from an impossible courage &mdash; <i>the Maid of Orl&eacute;ans, Warrior, Martyr, "
            "Visionary, France, Armour</i> &mdash; this portrait honours the farm girl who heard the "
            "voices of saints and led an army at seventeen.</p>"
            "<p>The Lord&rsquo;s charge to Josue was hers: &ldquo;Take courage, and be strong&hellip; "
            "because the Lord thy God is with thee&rdquo; (Josue 1:9). She crowned a king, was sold to "
            "her enemies, and was burned at nineteen with the name of Jesus on her lips &mdash; condemned "
            "by men and canonised by the Church. Close, it is courage in words; far, it is the face of a "
            "saint in armour.</p>"),
    },

    # --- Doctors & Martyrs --------------------------------------------------
    "st-augustine-of-hippo": {
        "scripture": "Not in rioting and drunkenness… but put ye on the Lord Jesus Christ.",
        "scripture_ref": "Romans 13:13–14 (D-R)",
        "about": (
            "<p>Made from a restless mind come home &mdash; <i>Bishop of Hippo, theologian, the "
            "Confessions, grace, conversion, Doctor of the Church</i> &mdash; this portrait honours the "
            "great sinner become the West&rsquo;s greatest teacher.</p>"
            "<p>In a garden he heard a child sing <i>take up and read</i>, and his eyes fell on the "
            "words: &ldquo;Not in rioting and drunkenness&hellip; but put ye on the Lord Jesus "
            "Christ&rdquo; (Romans 13:13&ndash;14). It was the end of his long flight and the beginning "
            "of the <i>Confessions</i>. <i>Late have I loved thee, Beauty ever ancient, ever new.</i> "
            "Close, it is grace in words; far, it is the face of a heart restless until it rested in "
            "God.</p>"),
    },
    "st-sebastian": {
        "scripture": "Labour as a good soldier of Christ Jesus.",
        "scripture_ref": "2 Timothy 2:3 (D-R)",
        "about": (
            "<p>Woven from a soldier&rsquo;s constancy &mdash; <i>Martyr, Roman soldier, the arrows, "
            "Courage, Resilience, patron of athletes</i> &mdash; this portrait honours the guardsman who "
            "kept his faith in Caesar&rsquo;s own household and paid for it with his life.</p>"
            "<p>Saint Paul&rsquo;s charge was his: &ldquo;Labour as a good soldier of Christ Jesus&rdquo; "
            "(2 Timothy 2:3). Shot through with arrows and left for dead, he survived to confront the "
            "emperor a second time. Patron of soldiers, athletes, and the sick, invoked against plague. "
            "Close, it is endurance in words; far, it is the face of a courage that would not break.</p>"),
    },
    "st-george": {
        "scripture": "Put you on the armour of God, that you may be able to stand against the deceits of the devil.",
        "scripture_ref": "Ephesians 6:11 (D-R)",
        "about": (
            "<p>Made from courage and legend both &mdash; <i>the dragon, Martyr, Knight, England, "
            "Chivalry, Protector</i> &mdash; this portrait honours the soldier-saint whose slaying of the "
            "dragon has stood for the conquest of evil ever since.</p>"
            "<p>The apostle&rsquo;s summons is his: &ldquo;Put you on the armour of God, that you may be "
            "able to stand against the deceits of the devil&rdquo; (Ephesians 6:11). A real martyr under "
            "Diocletian, remembered in a legend where faith unhorses the dragon. Patron of England, "
            "soldiers, and scouts. Close, it is spiritual armour in words; far, it is the face of the "
            "knight who rode at evil and won.</p>"),
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
