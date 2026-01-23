"""
Text normalization for TTS edge cases.

Handles conversion of special text patterns to speakable form:
- Numbers (large numbers, decimals, ordinals)
- Currency ($100 -> one hundred dollars)
- Abbreviations (Dr. -> Doctor)
- Acronyms (FBI -> F B I)
- Emojis (😀 -> smiling face)
- Math symbols (× -> times)
- URLs and emails
- HTML entities
"""

import re
import html
from typing import Optional

# Common abbreviations and their expansions
ABBREVIATIONS = {
    # Titles
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Prof.": "Professor",
    "Sr.": "Senior",
    "Jr.": "Junior",
    "Rev.": "Reverend",
    "Gen.": "General",
    "Col.": "Colonel",
    "Lt.": "Lieutenant",
    "Capt.": "Captain",
    "Sgt.": "Sergeant",

    # Common
    "St.": "Street",  # Context-dependent, could be Saint
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Rd.": "Road",
    "Drv.": "Drive",  # Use Drv. for Drive to avoid conflict with Dr. (Doctor)
    "Apt.": "Apartment",
    "Ste.": "Suite",
    "Bldg.": "Building",
    "Fl.": "Floor",

    # Units
    "ft.": "feet",
    "in.": "inches",
    "lb.": "pounds",
    "lbs.": "pounds",
    "oz.": "ounces",
    "pt.": "pint",
    "qt.": "quart",
    "gal.": "gallon",
    "mi.": "miles",
    "yd.": "yards",
    "km.": "kilometers",
    "cm.": "centimeters",
    "mm.": "millimeters",
    "kg.": "kilograms",
    "mg.": "milligrams",
    "ml.": "milliliters",

    # Time
    "min.": "minutes",
    "sec.": "seconds",
    "hr.": "hour",
    "hrs.": "hours",
    "wk.": "week",
    "mo.": "month",
    "yr.": "year",
    "yrs.": "years",

    # Other common
    "vs.": "versus",
    "etc.": "etcetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
    "govt.": "government",
    "dept.": "department",
    "est.": "established",
    "inc.": "incorporated",
    "corp.": "corporation",
    "ltd.": "limited",
    "assoc.": "association",
    "no.": "number",
    "vol.": "volume",
    "pg.": "page",
    "pp.": "pages",
    "ch.": "chapter",
    "fig.": "figure",
    "max.": "maximum",
    "min.": "minimum",
    "avg.": "average",
    "temp.": "temperature",
    "ref.": "reference",
    "info.": "information",
}

# Currency symbols and names
CURRENCY_SYMBOLS = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
    "₹": ("rupee", "rupees"),
    "₽": ("ruble", "rubles"),
    "₩": ("won", "won"),
    "฿": ("baht", "baht"),
    "₿": ("bitcoin", "bitcoins"),
    "Fr": ("franc", "francs"),
    "kr": ("krona", "kronor"),
    "R$": ("real", "reais"),
    "A$": ("australian dollar", "australian dollars"),
    "C$": ("canadian dollar", "canadian dollars"),
    "HK$": ("hong kong dollar", "hong kong dollars"),
    "NZ$": ("new zealand dollar", "new zealand dollars"),
    "S$": ("singapore dollar", "singapore dollars"),
    "CHF": ("swiss franc", "swiss francs"),
}

# Math symbols
MATH_SYMBOLS = {
    "+": "plus",
    "-": "minus",
    "×": "times",
    "÷": "divided by",
    "±": "plus or minus",
    "=": "equals",
    "≠": "not equal to",
    "≈": "approximately equal to",
    "<": "less than",
    ">": "greater than",
    "≤": "less than or equal to",
    "≥": "greater than or equal to",
    "%": "percent",
    "°": "degrees",
    "√": "square root of",
    "∞": "infinity",
    "π": "pi",
    "∑": "sum of",
    "∏": "product of",
    "∫": "integral of",
    "∂": "partial",
    "Δ": "delta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "σ": "sigma",
    "ω": "omega",
}

# Common emoji mappings (subset of most used)
EMOJI_NAMES = {
    "😀": "grinning face",
    "😃": "smiling face with big eyes",
    "😄": "smiling face with smiling eyes",
    "😁": "beaming face",
    "😆": "laughing face",
    "😅": "grinning face with sweat",
    "🤣": "rolling on floor laughing",
    "😂": "face with tears of joy",
    "🙂": "slightly smiling face",
    "😊": "smiling face with smiling eyes",
    "😇": "smiling face with halo",
    "🥰": "smiling face with hearts",
    "😍": "heart eyes",
    "🤩": "star struck",
    "😘": "face blowing kiss",
    "😗": "kissing face",
    "😚": "kissing face with closed eyes",
    "😋": "face savoring food",
    "😛": "face with tongue",
    "😜": "winking face with tongue",
    "🤪": "zany face",
    "😝": "squinting face with tongue",
    "🤑": "money mouth face",
    "🤗": "hugging face",
    "🤭": "face with hand over mouth",
    "🤫": "shushing face",
    "🤔": "thinking face",
    "🤐": "zipper mouth face",
    "🤨": "face with raised eyebrow",
    "😐": "neutral face",
    "😑": "expressionless face",
    "😶": "face without mouth",
    "😏": "smirking face",
    "😒": "unamused face",
    "🙄": "face with rolling eyes",
    "😬": "grimacing face",
    "🤥": "lying face",
    "😌": "relieved face",
    "😔": "pensive face",
    "😪": "sleepy face",
    "🤤": "drooling face",
    "😴": "sleeping face",
    "😷": "face with medical mask",
    "🤒": "face with thermometer",
    "🤕": "face with bandage",
    "🤢": "nauseated face",
    "🤮": "vomiting face",
    "🤧": "sneezing face",
    "🥵": "hot face",
    "🥶": "cold face",
    "🥴": "woozy face",
    "😵": "dizzy face",
    "🤯": "exploding head",
    "🤠": "cowboy hat face",
    "🥳": "partying face",
    "😎": "smiling face with sunglasses",
    "🤓": "nerd face",
    "🧐": "face with monocle",
    "😕": "confused face",
    "😟": "worried face",
    "🙁": "slightly frowning face",
    "😮": "face with open mouth",
    "😯": "hushed face",
    "😲": "astonished face",
    "😳": "flushed face",
    "🥺": "pleading face",
    "😦": "frowning face with open mouth",
    "😧": "anguished face",
    "😨": "fearful face",
    "😰": "anxious face with sweat",
    "😥": "sad but relieved face",
    "😢": "crying face",
    "😭": "loudly crying face",
    "😱": "face screaming in fear",
    "😖": "confounded face",
    "😣": "persevering face",
    "😞": "disappointed face",
    "😓": "downcast face with sweat",
    "😩": "weary face",
    "😫": "tired face",
    "🥱": "yawning face",
    "😤": "face with steam from nose",
    "😡": "pouting face",
    "😠": "angry face",
    "🤬": "face with symbols on mouth",
    "😈": "smiling face with horns",
    "👿": "angry face with horns",
    "💀": "skull",
    "☠️": "skull and crossbones",
    "💩": "pile of poo",
    "🤡": "clown face",
    "👹": "ogre",
    "👺": "goblin",
    "👻": "ghost",
    "👽": "alien",
    "👾": "alien monster",
    "🤖": "robot",
    "😺": "smiling cat",
    "😸": "grinning cat",
    "😹": "cat with tears of joy",
    "😻": "smiling cat with heart eyes",
    "😼": "cat with wry smile",
    "😽": "kissing cat",
    "🙀": "weary cat",
    "😿": "crying cat",
    "😾": "pouting cat",
    "🙈": "see no evil monkey",
    "🙉": "hear no evil monkey",
    "🙊": "speak no evil monkey",
    "❤️": "red heart",
    "🧡": "orange heart",
    "💛": "yellow heart",
    "💚": "green heart",
    "💙": "blue heart",
    "💜": "purple heart",
    "🖤": "black heart",
    "🤍": "white heart",
    "🤎": "brown heart",
    "💔": "broken heart",
    "💕": "two hearts",
    "💞": "revolving hearts",
    "💓": "beating heart",
    "💗": "growing heart",
    "💖": "sparkling heart",
    "💘": "heart with arrow",
    "💝": "heart with ribbon",
    "👍": "thumbs up",
    "👎": "thumbs down",
    "👋": "waving hand",
    "🤚": "raised back of hand",
    "🖐️": "hand with fingers splayed",
    "✋": "raised hand",
    "🖖": "vulcan salute",
    "👌": "OK hand",
    "🤌": "pinched fingers",
    "🤏": "pinching hand",
    "✌️": "victory hand",
    "🤞": "crossed fingers",
    "🤟": "love you gesture",
    "🤘": "sign of the horns",
    "🤙": "call me hand",
    "👈": "backhand index pointing left",
    "👉": "backhand index pointing right",
    "👆": "backhand index pointing up",
    "👇": "backhand index pointing down",
    "☝️": "index pointing up",
    "👏": "clapping hands",
    "🙌": "raising hands",
    "👐": "open hands",
    "🤲": "palms up together",
    "🤝": "handshake",
    "🙏": "folded hands",
    "✍️": "writing hand",
    "💪": "flexed biceps",
    "🦾": "mechanical arm",
    "🦿": "mechanical leg",
    "🦵": "leg",
    "🦶": "foot",
    "👂": "ear",
    "🦻": "ear with hearing aid",
    "👃": "nose",
    "🧠": "brain",
    "👀": "eyes",
    "👁️": "eye",
    "👅": "tongue",
    "👄": "mouth",
    "🎉": "party popper",
    "🎊": "confetti ball",
    "🎈": "balloon",
    "🎁": "wrapped gift",
    "🎂": "birthday cake",
    "🎄": "christmas tree",
    "🎃": "jack o lantern",
    "✨": "sparkles",
    "⭐": "star",
    "🌟": "glowing star",
    "💫": "dizzy",
    "🔥": "fire",
    "💥": "collision",
    "💦": "sweat droplets",
    "💨": "dashing away",
    "🌈": "rainbow",
    "☀️": "sun",
    "🌤️": "sun behind small cloud",
    "⛅": "sun behind cloud",
    "🌥️": "sun behind large cloud",
    "☁️": "cloud",
    "🌧️": "cloud with rain",
    "⛈️": "cloud with lightning and rain",
    "🌩️": "cloud with lightning",
    "🌨️": "cloud with snow",
    "❄️": "snowflake",
    "☃️": "snowman",
    "⛄": "snowman without snow",
    "🌪️": "tornado",
    "🌫️": "fog",
    "🌊": "water wave",
    "💧": "droplet",
    "☔": "umbrella with rain drops",
    "⚡": "high voltage",
    "🍎": "red apple",
    "🍕": "pizza",
    "🍔": "hamburger",
    "🍟": "french fries",
    "🌭": "hot dog",
    "🍿": "popcorn",
    "🍩": "doughnut",
    "🍪": "cookie",
    "🎵": "musical note",
    "🎶": "musical notes",
    "🎤": "microphone",
    "🎧": "headphone",
    "🎸": "guitar",
    "🎹": "musical keyboard",
    "🎺": "trumpet",
    "🎻": "violin",
    "🥁": "drum",
    "📱": "mobile phone",
    "💻": "laptop",
    "🖥️": "desktop computer",
    "⌨️": "keyboard",
    "🖱️": "computer mouse",
    "💾": "floppy disk",
    "💿": "optical disk",
    "📀": "dvd",
    "🔌": "electric plug",
    "💡": "light bulb",
    "🔦": "flashlight",
    "🔋": "battery",
    "✅": "check mark button",
    "❌": "cross mark",
    "❓": "question mark",
    "❗": "exclamation mark",
    "⚠️": "warning",
    "🚫": "prohibited",
    "⛔": "no entry",
    "🔴": "red circle",
    "🟠": "orange circle",
    "🟡": "yellow circle",
    "🟢": "green circle",
    "🔵": "blue circle",
    "🟣": "purple circle",
    "⚫": "black circle",
    "⚪": "white circle",
    "🟤": "brown circle",
}

# Number words
ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]


def number_to_words(n: int) -> str:
    """Convert an integer to words.

    Args:
        n: Integer to convert (supports up to quadrillions)

    Returns:
        String representation in words
    """
    if n == 0:
        return "zero"

    if n < 0:
        return "negative " + number_to_words(-n)

    if n < 20:
        return ONES[n]

    if n < 100:
        tens, ones = divmod(n, 10)
        return TENS[tens] + ("-" + ONES[ones] if ones else "")

    if n < 1000:
        hundreds, remainder = divmod(n, 100)
        result = ONES[hundreds] + " hundred"
        if remainder:
            result += " " + number_to_words(remainder)
        return result

    # Handle thousands, millions, billions, etc.
    parts = []
    scale_index = 0

    while n > 0:
        n, chunk = divmod(n, 1000)
        if chunk:
            chunk_words = number_to_words(chunk) if chunk < 1000 else str(chunk)
            if scale_index > 0:
                chunk_words += " " + SCALES[scale_index]
            parts.append(chunk_words)
        scale_index += 1

    return " ".join(reversed(parts))


def decimal_to_words(value: str) -> str:
    """Convert a decimal number string to words.

    Args:
        value: Decimal string like "3.14" or "0.5"

    Returns:
        String representation like "three point one four"
    """
    if "." not in value:
        try:
            return number_to_words(int(value))
        except ValueError:
            return value

    parts = value.split(".")
    if len(parts) != 2:
        return value

    integer_part, decimal_part = parts

    # Handle integer part
    try:
        int_val = int(integer_part) if integer_part else 0
        result = number_to_words(int_val)
    except ValueError:
        result = integer_part

    # Handle decimal part - read digit by digit
    result += " point"
    for digit in decimal_part:
        if digit.isdigit():
            result += " " + ONES[int(digit)] if int(digit) > 0 else " zero"
        else:
            result += " " + digit

    return result


def expand_currency(text: str) -> str:
    """Expand currency symbols to words.

    Args:
        text: Text potentially containing currency amounts

    Returns:
        Text with currency expanded (e.g., "$100" -> "one hundred dollars")
    """
    result = text

    # Pattern for currency: symbol followed by number (with optional commas/decimals)
    for symbol, (singular, plural) in CURRENCY_SYMBOLS.items():
        # Escape special regex characters
        escaped_symbol = re.escape(symbol)
        pattern = rf'{escaped_symbol}\s*([\d,]+(?:\.\d+)?)'

        def replace_currency(match):
            amount_str = match.group(1).replace(",", "")
            try:
                if "." in amount_str:
                    # Handle cents
                    parts = amount_str.split(".")
                    dollars = int(parts[0])
                    cents = int(parts[1][:2].ljust(2, '0'))  # Pad to 2 digits

                    dollar_word = number_to_words(dollars)
                    dollar_unit = singular if dollars == 1 else plural

                    if cents > 0:
                        cent_word = number_to_words(cents)
                        cent_unit = "cent" if cents == 1 else "cents"
                        return f"{dollar_word} {dollar_unit} and {cent_word} {cent_unit}"
                    else:
                        return f"{dollar_word} {dollar_unit}"
                else:
                    amount = int(amount_str)
                    word = number_to_words(amount)
                    unit = singular if amount == 1 else plural
                    return f"{word} {unit}"
            except ValueError:
                return match.group(0)

        result = re.sub(pattern, replace_currency, result)

    return result


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations to full words.

    Args:
        text: Text containing abbreviations

    Returns:
        Text with abbreviations expanded
    """
    result = text

    # Sort by length (longest first) to avoid partial replacements
    sorted_abbrevs = sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0]))

    for abbrev, expansion in sorted_abbrevs:
        # Use word boundary matching
        pattern = rf'\b{re.escape(abbrev)}'
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

    return result


def expand_acronyms(text: str) -> str:
    """Expand acronyms to be spelled letter by letter.

    Args:
        text: Text containing acronyms

    Returns:
        Text with acronyms spelled out (e.g., "FBI" -> "F B I")
    """
    # Common acronyms that should be spelled out
    spell_out = {
        "FBI", "CIA", "NSA", "NASA", "USA", "UK", "EU", "UN", "CEO", "CFO",
        "CTO", "COO", "HR", "IT", "AI", "ML", "API", "URL", "HTML", "CSS",
        "JSON", "XML", "SQL", "PDF", "GPS", "ATM", "PIN", "USB", "HDMI",
        "CPU", "GPU", "RAM", "ROM", "SSD", "HDD", "LED", "LCD", "TV", "DVD",
        "CD", "MP3", "MP4", "JPG", "PNG", "GIF", "ZIP", "EXE", "DLL", "SDK",
        "IDE", "VPN", "LAN", "WAN", "WiFi", "HTTP", "HTTPS", "FTP", "SSH",
        "DNS", "IP", "TCP", "UDP", "iOS", "macOS", "FAQ", "DIY", "ASAP",
        "FYI", "BTW", "OMG", "LOL", "BRB", "IMO", "TBD", "TBA", "AKA",
        "ETA", "RSVP", "PS", "NB", "CV", "ID", "VIP", "RIP", "MIA", "POW",
        "AWOL", "SWAT", "SCUBA", "RADAR", "LASER", "NATO", "OPEC", "UNICEF",
    }

    result = text

    for acronym in spell_out:
        # Match whole word only
        pattern = rf'\b{re.escape(acronym)}\b'
        spaced = " ".join(acronym.upper())
        result = re.sub(pattern, spaced, result, flags=re.IGNORECASE)

    return result


def expand_emojis(text: str) -> str:
    """Convert emojis to their text descriptions.

    Args:
        text: Text containing emojis

    Returns:
        Text with emojis replaced by descriptions
    """
    result = text

    for emoji, name in EMOJI_NAMES.items():
        result = result.replace(emoji, f" {name} ")

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)

    return result.strip()


def expand_math_symbols(text: str) -> str:
    """Expand mathematical symbols to words.

    Args:
        text: Text containing math symbols

    Returns:
        Text with symbols expanded (e.g., "2 + 2 = 4" -> "2 plus 2 equals 4")
    """
    result = text

    for symbol, word in MATH_SYMBOLS.items():
        result = result.replace(symbol, f" {word} ")

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)

    return result.strip()


def decode_html_entities(text: str) -> str:
    """Decode HTML entities to their character equivalents.

    Args:
        text: Text containing HTML entities

    Returns:
        Text with entities decoded
    """
    return html.unescape(text)


def expand_url(url: str) -> str:
    """Convert a URL to speakable form.

    Args:
        url: URL string

    Returns:
        Speakable representation
    """
    # Remove protocol
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)

    # Replace special characters
    url = url.replace('.', ' dot ')
    url = url.replace('/', ' slash ')
    url = url.replace('-', ' dash ')
    url = url.replace('_', ' underscore ')
    url = url.replace('@', ' at ')
    url = url.replace('?', ' question mark ')
    url = url.replace('&', ' and ')
    url = url.replace('=', ' equals ')

    return url.strip()


def expand_email(email: str) -> str:
    """Convert an email address to speakable form.

    Args:
        email: Email address

    Returns:
        Speakable representation
    """
    email = email.replace('@', ' at ')
    email = email.replace('.', ' dot ')
    email = email.replace('-', ' dash ')
    email = email.replace('_', ' underscore ')

    return email.strip()


def expand_urls_and_emails(text: str) -> str:
    """Find and expand URLs and emails in text.

    Args:
        text: Text potentially containing URLs or emails

    Returns:
        Text with URLs and emails expanded
    """
    # URL pattern
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'

    def replace_url(match):
        return expand_url(match.group(0))

    result = re.sub(url_pattern, replace_url, text)

    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    def replace_email(match):
        return expand_email(match.group(0))

    result = re.sub(email_pattern, replace_email, result)

    return result


def normalize_text(
    text: str,
    expand_numbers_flag: bool = True,
    expand_currency_flag: bool = True,
    expand_abbreviations_flag: bool = True,
    expand_acronyms_flag: bool = False,  # Can make text longer, off by default
    expand_emojis_flag: bool = True,
    expand_math_flag: bool = True,
    expand_urls_flag: bool = True,
    decode_html_flag: bool = True,
) -> str:
    """Apply all text normalizations for TTS.

    Args:
        text: Input text to normalize
        expand_numbers_flag: Expand large numbers to words
        expand_currency_flag: Expand currency amounts
        expand_abbreviations_flag: Expand common abbreviations
        expand_acronyms_flag: Spell out acronyms letter by letter
        expand_emojis_flag: Convert emojis to text
        expand_math_flag: Expand math symbols
        expand_urls_flag: Expand URLs and emails
        decode_html_flag: Decode HTML entities

    Returns:
        Normalized text ready for TTS
    """
    result = text

    # Decode HTML entities first
    if decode_html_flag:
        result = decode_html_entities(result)

    # Expand currency (before general number expansion)
    if expand_currency_flag:
        result = expand_currency(result)

    # Expand emojis
    if expand_emojis_flag:
        result = expand_emojis(result)

    # Expand math symbols
    if expand_math_flag:
        result = expand_math_symbols(result)

    # Expand URLs and emails
    if expand_urls_flag:
        result = expand_urls_and_emails(result)

    # Expand abbreviations
    if expand_abbreviations_flag:
        result = expand_abbreviations(result)

    # Expand acronyms (optional, can make text verbose)
    if expand_acronyms_flag:
        result = expand_acronyms(result)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result
