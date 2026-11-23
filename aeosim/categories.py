"""Service categories the simulator can generate markets for.

Each category lists the services firms offer (with the words a buyer or a
website might use for them), the certifications that matter, the regions
firms operate in, and the buyer types that shop for the service. Adding a
category means adding one entry to CATEGORIES.
"""

REGIONS = {
    "Midwest": {
        "Ohio": ["Columbus", "Cleveland", "Dayton", "Toledo"],
        "Indiana": ["Indianapolis", "Fort Wayne"],
        "Michigan": ["Detroit", "Grand Rapids"],
        "Illinois": ["Chicago", "Joliet", "Peoria"],
    },
    "Northeast": {
        "New Jersey": ["Newark", "Edison", "Trenton"],
        "Pennsylvania": ["Allentown", "Philadelphia", "Pittsburgh"],
        "New York": ["Buffalo", "Syracuse"],
    },
    "Southeast": {
        "Georgia": ["Atlanta", "Savannah"],
        "North Carolina": ["Charlotte", "Greensboro"],
        "Tennessee": ["Memphis", "Nashville"],
    },
    "Texas": {
        "Texas": ["Dallas", "Houston", "Laredo", "San Antonio"],
    },
}

NEIGHBORS = {
    "Midwest": ["Northeast", "Southeast"],
    "Northeast": ["Midwest"],
    "Southeast": ["Midwest", "Texas"],
    "Texas": ["Southeast"],
}

CATEGORIES = {
    "freight_trucking": {
        "label": "freight trucking company",
        "provider_noun": "trucking company",
        "name_nouns": ["Freight", "Logistics", "Transport", "Trucking", "Carriers", "Haulers", "Freight Lines"],
        "services": {
            "refrigerated": {
                "label": "refrigerated trucking",
                "terms": ["refrigerated", "reefer", "temperature-controlled", "cold chain"],
                "buyer_need": "refrigerated loads",
            },
            "flatbed": {
                "label": "flatbed hauling",
                "terms": ["flatbed", "step deck", "oversize"],
                "buyer_need": "flatbed loads of steel and machinery",
            },
            "ltl": {
                "label": "LTL (less-than-truckload) shipping",
                "terms": ["LTL", "less-than-truckload", "partial loads", "pallet shipping"],
                "buyer_need": "partial loads of a few pallets at a time",
            },
            "full_truckload": {
                "label": "full truckload shipping",
                "terms": ["full truckload", "FTL", "dry van", "53-foot trailers"],
                "buyer_need": "full truckloads in dry vans",
            },
            "hazmat": {
                "label": "hazmat transport",
                "terms": ["hazmat", "hazardous materials", "chemical transport"],
                "buyer_need": "hazardous materials like industrial solvents",
            },
            "drayage": {
                "label": "port drayage",
                "terms": ["drayage", "container hauling", "port", "intermodal"],
                "buyer_need": "containers moved from the port to our warehouse",
            },
            "warehousing": {
                "label": "warehousing and cross-docking",
                "terms": ["warehousing", "cross-docking", "storage", "distribution center"],
                "buyer_need": "short-term storage and cross-docking",
            },
            "expedited": {
                "label": "expedited freight",
                "terms": ["expedited", "time-critical", "hot shot", "same-day"],
                "buyer_need": "time-critical shipments that cannot wait",
            },
        },
        "certifications": {
            "smartway": {"label": "EPA SmartWay Partner", "terms": ["SmartWay"]},
            "fsma": {"label": "FSMA Sanitary Transportation compliant", "terms": ["FSMA", "sanitary transportation", "food safety"]},
            "hazmat_cert": {"label": "USDOT Hazmat registered", "terms": ["hazmat registered", "PHMSA", "USDOT hazmat"]},
            "ctpat": {"label": "C-TPAT certified", "terms": ["C-TPAT", "customs-trade partnership"]},
            "iso9001": {"label": "ISO 9001 certified", "terms": ["ISO 9001", "quality management"]},
        },
        "service_cert_links": {"refrigerated": "fsma", "hazmat": "hazmat_cert", "drayage": "ctpat"},
        "buyers": [
            ("food manufacturer", "refrigerated"),
            ("steel fabricator", "flatbed"),
            ("furniture maker", "ltl"),
            ("consumer goods distributor", "full_truckload"),
            ("specialty chemicals plant", "hazmat"),
            ("electronics importer", "drayage"),
            ("e-commerce brand", "warehousing"),
            ("auto parts supplier", "expedited"),
        ],
        "volume_phrases": [
            "about {n} shipments a week",
            "roughly {n} loads a month",
            "{n} pickups a week",
        ],
        "size_unit": "trucks",
        "size_range": (8, 400),
    },
    "contract_packaging": {
        "label": "contract packaging company",
        "provider_noun": "contract packager",
        "name_nouns": ["Packaging", "Pack", "Co-Packing", "Fulfillment", "Packaging Solutions"],
        "services": {
            "co_packing_food": {
                "label": "food co-packing",
                "terms": ["co-packing", "co-packer", "food packaging", "copacking"],
                "buyer_need": "co-packing for a new snack product",
            },
            "blister": {
                "label": "blister and clamshell packaging",
                "terms": ["blister", "clamshell", "thermoforming"],
                "buyer_need": "blister packs for small consumer products",
            },
            "kitting": {
                "label": "kitting and assembly",
                "terms": ["kitting", "assembly", "subscription boxes", "gift sets"],
                "buyer_need": "kitting for subscription boxes",
            },
            "shrink": {
                "label": "shrink wrapping and bundling",
                "terms": ["shrink wrap", "shrink sleeve", "bundling", "multipacks"],
                "buyer_need": "shrink-wrapped multipacks",
            },
            "liquid_fill": {
                "label": "liquid filling",
                "terms": ["liquid filling", "bottling", "bottle filling", "lotions"],
                "buyer_need": "liquid filling for lotions and cleaners",
            },
            "pouch_fill": {
                "label": "pouch filling",
                "terms": ["pouch filling", "stand-up pouches", "sachets", "stick packs"],
                "buyer_need": "stand-up pouch filling for powders",
            },
            "display": {
                "label": "retail display assembly",
                "terms": ["display assembly", "point-of-purchase", "POP displays", "retail displays"],
                "buyer_need": "assembly of retail floor displays",
            },
        },
        "certifications": {
            "sqf": {"label": "SQF Level 2 certified", "terms": ["SQF", "safe quality food"]},
            "fda": {"label": "FDA registered facility", "terms": ["FDA registered", "FDA"]},
            "gmp": {"label": "cGMP compliant", "terms": ["cGMP", "GMP", "good manufacturing practice"]},
            "organic": {"label": "USDA Organic certified handler", "terms": ["organic", "USDA organic"]},
            "iso9001": {"label": "ISO 9001 certified", "terms": ["ISO 9001", "quality management"]},
        },
        "service_cert_links": {"co_packing_food": "sqf", "pouch_fill": "fda", "liquid_fill": "gmp"},
        "buyers": [
            ("snack food startup", "co_packing_food"),
            ("consumer electronics accessory brand", "blister"),
            ("subscription box company", "kitting"),
            ("beverage brand", "shrink"),
            ("personal care brand", "liquid_fill"),
            ("supplement company", "pouch_fill"),
            ("toy company", "display"),
        ],
        "volume_phrases": [
            "about {n} thousand units a month",
            "a first run of {n} thousand units",
            "{n} thousand units a quarter",
        ],
        "size_unit": "employees",
        "size_range": (10, 600),
    },
}
