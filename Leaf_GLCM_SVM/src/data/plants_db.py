# src/data/plants_db.py
# ============================================================
# LeafVision AI — Plants Database v2.0
# Swedish Leaf Dataset — 15 loài (Label 0–14)
# Đã được Việt hóa 100% giá trị và tinh chỉnh mô tả sát với ảnh Dataset.
# ============================================================

PLANTS_DB = {

    # =========================================================
    # 0 — Ulmus carpinifolia
    # =========================================================
    "0": {
        "name_vn":          "Cây Du đồng",
        "scientific_name":  "Ulmus carpinifolia",
        "english_name":     "Field Elm",
        "common_names":     ["Du đồng", "Du lá nhẵn"],

        "family":           "Họ Du",
        "genus":            "Chi Du",
        "origin":           "Châu Âu",

        "habitat":
            "Vùng đồng bằng, ven rừng và khu vực khí hậu ôn đới châu Âu.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình bầu dục đến hình trứng",
        "leaf_margin":      "Răng cưa kép",
        "leaf_texture":     "Hơi nhám",
        "vein_pattern":     "Gân lông chim",

        "description":
            "Lá có mép khía răng cưa kép. Đặc điểm nhận diện cốt lõi trong dataset: gốc lá bất đối xứng rõ rệt (một bên thấp, một bên cao).",

        "medical_uses":
            "Hỗ trợ chữa lành vết thương ngoài da.",

        "toxicity":             "Không độc",
        "average_height":       "20 - 35m",
        "lifespan":             "100 - 250 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Trung bình",
            "edge_complexity":  "Cao",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 1 — Acer (Maple)
    # =========================================================
    "1": {
        "name_vn":          "Cây Phong",
        "scientific_name":  "Acer",
        "english_name":     "Maple",
        "common_names":     ["Phong", "Thích"],

        "family":           "Họ Bồ hòn",
        "genus":            "Chi Phong",
        "origin":           "Bắc bán cầu",

        "habitat":
            "Vùng ôn đới Bắc bán cầu, phổ biến ở rừng rụng lá Thụy Điển.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Xẻ thùy chân vịt",
        "leaf_margin":      "Có thùy nhọn",
        "leaf_texture":     "Nhẵn mịn",
        "vein_pattern":     "Gân hình mạng tỏa tròn",

        "description":
            "Hình dáng xẻ thùy chân vịt vô cùng đặc trưng, thường có 5 thùy nhọn to, mép có răng cưa lác đác. Dễ nhận diện nhất đối với AI.",

        "medical_uses":
            "Làm thuốc thanh nhiệt, nhựa dùng làm si-rô.",

        "toxicity":             "Không độc",
        "average_height":       "10 - 45m",
        "lifespan":             "80 - 300 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Rất cao",
            "vein_density":     "Cao"
        }
    },

    # =========================================================
    # 2 — Salix aurita
    # =========================================================
    "2": {
        "name_vn":          "Cây Liễu tai chuột",
        "scientific_name":  "Salix aurita",
        "english_name":     "Eared Willow",
        "common_names":     ["Liễu tai chuột", "Liễu tai"],

        "family":           "Họ Liễu",
        "genus":            "Chi Liễu",
        "origin":           "Châu Âu",

        "habitat":
            "Vùng đất ẩm, bụi rậm, ven đầm lầy tại Bắc Âu.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình bầu dục ngược hoặc thuôn nhỏ",
        "leaf_margin":      "Lượn sóng hoặc răng cưa nhẹ",
        "leaf_texture":     "Nhăn nheo, mặt dưới có lông tơ",
        "vein_pattern":     "Gân lông chim",

        "description":
            "Lá nhỏ, bề mặt có độ nhăn nhẹ (texture gồ ghề). Chóp lá có một mũi nhọn nhỏ hơi vặn xoắn. Dễ nhầm với Liễu xám nhưng kích thước nhỏ hơn.",

        "medical_uses":
            "Vỏ chứa salicin dùng để giảm đau, hạ sốt tự nhiên.",

        "toxicity":             "Không độc",
        "average_height":       "2 - 6m",
        "lifespan":             "40 - 80 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Cao",
            "edge_complexity":  "Trung bình",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 3 — Quercus (Oak)
    # =========================================================
    "3": {
        "name_vn":          "Cây Sồi",
        "scientific_name":  "Quercus",
        "english_name":     "Oak",
        "common_names":     ["Sồi", "Sồi Anh"],

        "family":           "Họ Cử",
        "genus":            "Chi Sồi",
        "origin":           "Bắc bán cầu",

        "habitat":
            "Rừng ôn đới, phát triển mạnh ở vùng nam và trung Thụy Điển.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Xẻ thùy lông chim sâu",
        "leaf_margin":      "Các thùy lượn tròn nhẵn",
        "leaf_texture":     "Dày, bề mặt bóng mờ",
        "vein_pattern":     "Gân nổi rõ ở mặt dưới",

        "description":
            "Mép lá lượn thùy rất sâu và phân bố không đều đặn, đỉnh các thùy luôn bo tròn. Rất dễ phân biệt dựa trên đường viền biên (edge complexity).",

        "medical_uses":
            "Vỏ chứa nhiều tannin, dùng để cầm máu và chữa bệnh tiêu hóa.",

        "toxicity":             "Độc nhẹ đối với gia súc nếu ăn nhiều quả sồi",
        "average_height":       "20 - 40m",
        "lifespan":             "200 - 1000 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Trung bình",
            "edge_complexity":  "Cao",
            "vein_density":     "Cao"
        }
    },

    # =========================================================
    # 4 — Alnus incana
    # =========================================================
    "4": {
        "name_vn":          "Cây Tống quán sủ xám",
        "scientific_name":  "Alnus incana",
        "english_name":     "Grey Alder",
        "common_names":     ["Tống quán sủ xám", "Trăn xám"],

        "family":           "Họ Cáng lò",
        "genus":            "Chi Tống quán sủ",
        "origin":           "Bắc bán cầu",

        "habitat":
            "Mọc thành thảm ven sông suối hoặc đất ẩm ướt, lạnh.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình xoan rộng, chóp nhọn",
        "leaf_margin":      "Răng cưa kép sắc nét",
        "leaf_texture":     "Mịn, hơi xám nhạt mặt dưới",
        "vein_pattern":     "Gân lông chim thẳng, lõm sâu",

        "description":
            "Lá hình trứng/xoan rộng, mép lá có hệ thống răng cưa kép sắc sảo. Các đường gân thẳng và song song với nhau rất đều đặn.",

        "medical_uses":
            "Có đặc tính kháng viêm, sắc uống chữa sưng tấy.",

        "toxicity":             "Không độc",
        "average_height":       "15 - 25m",
        "lifespan":             "60 - 100 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Trung bình",
            "edge_complexity":  "Cao",
            "vein_density":     "Cao"
        }
    },

    # =========================================================
    # 5 — Betula pubescens
    # =========================================================
    "5": {
        "name_vn":          "Cây Bạch dương lông",
        "scientific_name":  "Betula pubescens",
        "english_name":     "Downy Birch",
        "common_names":     ["Bạch dương lông", "Bạch dương trắng"],

        "family":           "Họ Bạch dương",
        "genus":            "Chi Bạch dương",
        "origin":           "Bắc Âu",

        "habitat":
            "Loài cây cực kỳ phổ biến tại Thụy Điển, sống trên đất than bùn, đầm lầy ẩm.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình trứng đến thoi",
        "leaf_margin":      "Răng cưa đơn, đều",
        "leaf_texture":     "Mỏng, nhẵn mặt trên",
        "vein_pattern":     "Gân lông chim",

        "description":
            "Lá hình trứng bo tròn ở gốc và thuôn nhọn ở chóp. Mép có răng cưa khá đều đặn. Nhìn trong dataset có hình khối tam giác cong.",

        "medical_uses":
            "Lợi tiểu, thanh lọc máu, hỗ trợ viêm đường tiết niệu.",

        "toxicity":             "Không độc",
        "average_height":       "10 - 25m",
        "lifespan":             "80 - 120 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Trung bình",
            "vein_density":     "Thấp"
        }
    },

    # =========================================================
    # 6 — Salix alba 'Tristis'
    # =========================================================
    "6": {
        "name_vn":          "Cây Liễu rủ",
        "scientific_name":  "Salix alba 'Tristis'",
        "english_name":     "Weeping Willow",
        "common_names":     ["Liễu rủ", "Liễu khóc"],

        "family":           "Họ Liễu",
        "genus":            "Chi Liễu",
        "origin":           "Châu Âu",

        "habitat":
            "Trồng phổ biến làm cảnh ven hồ, sông ngòi khu vực ôn đới.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình mác hẹp, thon dài",
        "leaf_margin":      "Răng cưa rất nhỏ và mịn",
        "leaf_texture":     "Mềm, trơn",
        "vein_pattern":     "Gân giữa nổi bật, gân phụ mờ",

        "description":
            "Lá rất hẹp và kéo dài, gần như một đường thẳng vuốt nhọn dần về chóp. Dạng hình học mảnh mai này AI phân biệt bằng tỷ lệ Dài/Rộng cực lớn.",

        "medical_uses":
            "Giảm đau tự nhiên, thành phần tương tự aspirin.",

        "toxicity":             "Không độc",
        "average_height":       "20 - 30m",
        "lifespan":             "40 - 75 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Thấp",
            "vein_density":     "Thấp"
        }
    },

    # =========================================================
    # 7 — Populus tremula
    # =========================================================
    "7": {
        "name_vn":          "Cây Bạch dương rung",
        "scientific_name":  "Populus tremula",
        "english_name":     "Aspen",
        "common_names":     ["Bạch dương rung", "Dương lá rung"],

        "family":           "Họ Liễu",
        "genus":            "Chi Dương",
        "origin":           "Châu Âu và châu Á",

        "habitat":
            "Rừng ôn đới, đặc biệt là các dải rừng xen kẽ đồng cỏ.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Gần tròn",
        "leaf_margin":      "Lượn sóng lồi lõm tròn to",
        "leaf_texture":     "Trơn nhẵn, mỏng",
        "vein_pattern":     "Gân hình mạng chân vịt",

        "description":
            "Đặc điểm nhận diện mạnh nhất trong tập dữ liệu: lá hình nón gần tròn, mép lượn sóng to không tạo thành khía sắc. Cuống dẹp dài (dù có thể đã bị cắt bớt trong ảnh scan).",

        "medical_uses":
            "Chữa viêm khớp và các bệnh liên quan đến nhiễm trùng bàng quang.",

        "toxicity":             "Không độc",
        "average_height":       "15 - 25m",
        "lifespan":             "60 - 100 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Trung bình",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 8 — Ulmus glabra
    # =========================================================
    "8": {
        "name_vn":          "Cây Du núi",
        "scientific_name":  "Ulmus glabra",
        "english_name":     "Wych Elm",
        "common_names":     ["Du núi", "Du Scotland"],

        "family":           "Họ Du",
        "genus":            "Chi Du",
        "origin":           "Châu Âu",

        "habitat":
            "Khu vực núi ôn đới Bắc Âu, thung lũng dốc và rừng sườn đồi.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình xoan ngược, to bản",
        "leaf_margin":      "Răng cưa kép nhọn sắc",
        "leaf_texture":     "Rất thô ráp (như giấy nhám)",
        "vein_pattern":     "Gân lông chim dày đặc, nổi cao",

        "description":
            "Lá có kích thước to, thường loe rộng ở nửa trên. Gốc lá lệch cực kỳ rõ rệt ôm lấy cuống ngắn. Mép có đôi khi phân 3 thùy nhọn ở phần chóp (tùy lá).",

        "medical_uses":
            "Làm dịu niêm mạc họng, hỗ trợ điều trị ho khan.",

        "toxicity":             "Không độc",
        "average_height":       "30 - 40m",
        "lifespan":             "150 - 300 năm",
        "conservation_status":  "Sắp bị đe dọa (do dịch bệnh nấm)",

        "ai_features": {
            "texture_level":    "Rất cao",
            "edge_complexity":  "Cao",
            "vein_density":     "Cao"
        }
    },

    # =========================================================
    # 9 — Sorbus aucuparia
    # =========================================================
    "9": {
        "name_vn":          "Cây Thanh lương trà",
        "scientific_name":  "Sorbus aucuparia",
        "english_name":     "Rowan",
        "common_names":     ["Thanh lương trà", "Hoa lê miền núi"],

        "family":           "Họ Hoa hồng",
        "genus":            "Chi Thanh lương trà",
        "origin":           "Châu Âu",

        "habitat":
            "Mọc rải rác ở đồi núi và bìa rừng lá rộng tại Scandinavia.",

        "leaf_type":        "Lá kép lông chim lẻ",
        "leaf_shape":       "Gồm nhiều lá chét thuôn hẹp",
        "leaf_margin":      "Răng cưa nhọn trên lá chét",
        "leaf_texture":     "Mịn mờ",
        "vein_pattern":     "Gân mờ không hiện rõ trên lá chét",

        "description":
            "Loài duy nhất trong Swedish dataset là dạng lá kép. Tuy nhiên trong ảnh scan, đôi khi nó là một cành chứa nhiều lá chét thuôn dài đối xứng, đỉnh có 1 lá chét chẵn.",

        "medical_uses":
            "Quả hái sau sương giá dùng bổ sung vitamin C, hỗ trợ hô hấp.",

        "toxicity":             "Không độc (nhưng hạt không nên ăn sống)",
        "average_height":       "5 - 15m",
        "lifespan":             "80 - 150 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Trung bình",
            "edge_complexity":  "Rất cao",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 10 — Salix cinerea
    # =========================================================
    "10": {
        "name_vn":          "Cây Liễu xám",
        "scientific_name":  "Salix cinerea",
        "english_name":     "Grey Willow",
        "common_names":     ["Liễu xám"],

        "family":           "Họ Liễu",
        "genus":            "Chi Liễu",
        "origin":           "Châu Âu",

        "habitat":
            "Vùng đất ngập nước, bãi than bùn và ao hồ Thụy Điển.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình xoan thuôn dài đến mác ngược",
        "leaf_margin":      "Gần nguyên hoặc gợn sóng cực nhẹ",
        "leaf_texture":     "Thô ráp, có lớp lông mượt",
        "vein_pattern":     "Gân lông chim lõm mặt trên",

        "description":
            "Dài và to hơn Liễu tai chuột. Nửa trên của lá phình to hơn nửa dưới (dạng mác ngược). Cấu trúc vân lá trên dataset hiển thị độ hạt (texture) cao do lông tơ dập nổi.",

        "medical_uses":
            "Thuốc hạ sốt tự nhiên cổ truyền.",

        "toxicity":             "Không độc",
        "average_height":       "4 - 10m",
        "lifespan":             "50 - 90 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Cao",
            "edge_complexity":  "Thấp",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 11 — Populus
    # =========================================================
    "11": {
        "name_vn":          "Cây Dương",
        "scientific_name":  "Populus",
        "english_name":     "Poplar",
        "common_names":     ["Dương đen", "Dương"],

        "family":           "Họ Liễu",
        "genus":            "Chi Dương",
        "origin":           "Bắc bán cầu",

        "habitat":
            "Vùng đất màu mỡ, ẩm thấp, hay được dùng làm rào chắn gió.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình nêm, thoi hoặc tam giác",
        "leaf_margin":      "Khía răng cưa tròn nhỏ",
        "leaf_texture":     "Nhẵn bóng",
        "vein_pattern":     "Gân tỏa chân vịt từ gốc",

        "description":
            "Hình dáng như một chiếc mũi giáo hoặc tam giác phình to ở phía dưới. Gốc lá thường là đường phẳng ngang vuông góc với cuống.",

        "medical_uses":
            "Chồi cây dùng chế tạo thuốc mỡ trị bỏng và đau khớp.",

        "toxicity":             "Không độc",
        "average_height":       "20 - 35m",
        "lifespan":             "80 - 150 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Trung bình",
            "vein_density":     "Trung bình"
        }
    },

    # =========================================================
    # 12 — Tilia
    # =========================================================
    "12": {
        "name_vn":          "Cây Đoạn",
        "scientific_name":  "Tilia",
        "english_name":     "Linden",
        "common_names":     ["Đoạn", "Bồ đề châu Âu"],

        "family":           "Họ Cẩm quỳ",
        "genus":            "Chi Đoạn",
        "origin":           "Bắc bán cầu",

        "habitat":
            "Rừng rụng lá hỗn giao, thường được trồng trên đường phố Bắc Âu.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình trái tim",
        "leaf_margin":      "Răng cưa nhọn sắc",
        "leaf_texture":     "Mềm, nhẵn mịn",
        "vein_pattern":     "Gân chân vịt mọc từ gốc",

        "description":
            "Khối lá hình trái tim (heart-shape) cực kỳ điển hình, chóp kéo dài thành đuôi nhọn. Gốc lá xẻ sâu tạo rãnh chữ V lệch.",

        "medical_uses":
            "Hoa pha trà giúp an thần, thư giãn và giảm triệu chứng cảm cúm.",

        "toxicity":             "Không độc",
        "average_height":       "20 - 40m",
        "lifespan":             "150 - 400 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Cao",
            "vein_density":     "Cao"
        }
    },

    # =========================================================
    # 13 — Sorbus intermedia
    # =========================================================
    "13": {
        "name_vn":          "Cây Thanh lương trà Thụy Điển",
        "scientific_name":  "Sorbus intermedia",
        "english_name":     "Swedish Whitebeam",
        "common_names":     ["Thanh lương trà Thụy Điển"],

        "family":           "Họ Hoa hồng",
        "genus":            "Chi Thanh lương trà",
        "origin":           "Nam Thụy Điển",

        "habitat":
            "Loài bản địa đặc hữu khu vực Baltic và nam Scandinavia.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Bầu dục to, xẻ thùy nông",
        "leaf_margin":      "Khía thùy nông và có răng cưa kép",
        "leaf_texture":     "Bề mặt có nếp gấp dọc gân",
        "vein_pattern":     "Gân lông chim song song thẳng tắp",

        "description":
            "Lá đơn (khác với Thanh lương trà thông thường là lá kép). Hai bên mép xẻ thành các thùy cạn dạng vỏ sò. Các gân phụ chạy cực kỳ thẳng tắp và song song.",

        "medical_uses":
            "Cung cấp dưỡng chất và hỗ trợ hệ thống miễn dịch nhẹ.",

        "toxicity":             "Không độc",
        "average_height":       "8 - 15m",
        "lifespan":             "80 - 150 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Trung bình",
            "edge_complexity":  "Cao",
            "vein_density":     "Rất cao"
        }
    },

    # =========================================================
    # 14 — Fagus sylvatica
    # =========================================================
    "14": {
        "name_vn":          "Cây Dẻ gai châu Âu",
        "scientific_name":  "Fagus sylvatica",
        "english_name":     "European Beech",
        "common_names":     ["Dẻ gai châu Âu", "Cử châu Âu"],

        "family":           "Họ Cử",
        "genus":            "Chi Dẻ gai",
        "origin":           "Châu Âu",

        "habitat":
            "Loài ưu thế ở các khu rừng bóng râm ôn đới trung tâm và tây Âu.",

        "leaf_type":        "Lá đơn",
        "leaf_shape":       "Hình xoan đều đặn",
        "leaf_margin":      "Nguyên hoặc gợn sóng thoai thoải",
        "leaf_texture":     "Trơn nhẵn, bóng",
        "vein_pattern":     "Gân song song chạy thẳng ra mép",

        "description":
            "Trong dataset, nó có viền ngoài gần như là một đường cong nhẵn hoàn hảo (edge complexity cực thấp). Gân song song chia bề mặt thành các rãnh chéo đều tăm tắp.",

        "medical_uses":
            "Nhựa và vỏ dùng sắc làm nước rửa chống viêm nhiễm ngoài da.",

        "toxicity":             "Không độc (nhưng hạt dẻ gai sống ăn nhiều có thể gây buồn nôn)",
        "average_height":       "30 - 40m",
        "lifespan":             "150 - 300 năm",
        "conservation_status":  "Ít quan tâm",

        "ai_features": {
            "texture_level":    "Thấp",
            "edge_complexity":  "Thấp",
            "vein_density":     "Cao"
        }
    },
}