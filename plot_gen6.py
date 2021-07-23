import cv2
from PIL import Image, ImageDraw, ImageFont
from os import listdir
from cfg_read import *
from numpy import array, load
import time


map_size, card_num, local_dir, db_dir, game_dir, output, skin_name, is_init = get_cfg()
song_folders = game_dir + '/music'
try:
    npy_path = local_dir + '/level_table.npy'
    level_table = load(npy_path)
except FileNotFoundError:
    input('level_table.npy not found, please check your file directory, '
          'unless this is the first time you have started the application.')
    pass

img_archive = local_dir + '/img_archive/gen6/'
if not path.exists(img_archive):
    input(r'Image archive is missing, please check your file directory.')
    sys.exit(1)

font_DFHS = img_archive + 'font/DF-HeiSeiMaruGothic-W4.ttc'
font_unipace = img_archive + 'unispace bd.ttf'
text_white = (245, 245, 245)
text_black = (61, 61, 61)
text_gray = (154, 154, 154)
clear_img = {'1': 'crash', '2': 'comp', '3': 'comp_ex', '4': 'uc', '5': 'puc'}
clear_table = {'1': 'FAILED', '2': 'NC', '3': 'HC', '4': 'UC', '5': 'PUC'}
grade_img = {'1': 'd', '2': 'c', '3': 'b', '4': 'a', '5': 'a_plus', '6': 'aa', '7': 'aa_plus', '8': 'aaa',
             '9': 'aaa_plus', '10': 's'}
grade_table = {'1': 'D', '2': 'C', '3': 'B', '4': 'A', '5': 'A+',
               '6': 'AA', '7': 'AA+', '8': 'AAA', '9': 'AAA+', '10': 'S'}


def get_vf_property(vf: float, is_darker: bool = False, is_level: bool = False) -> tuple or int:
    if vf < 0:
        level = 0
        color = text_white
    elif vf < 10:
        level = 1
        color = (193, 139, 73)
    elif vf < 12:
        level = 2
        color = (48, 73, 157)
    elif vf < 14:
        level = 3
        color = (245, 189, 26)
    elif vf < 15:
        level = 4
        color = (83, 189, 181)
    elif vf < 16:
        level = 5
        color = (200, 22, 30)
    elif vf < 17:
        level = 6
        color = (237, 179, 202)
    elif vf < 18:
        level = 7
        color = (234, 234, 233)
    elif vf < 19:
        level = 8
        color = (248, 227, 165)
    elif vf < 20:
        level = 9
        color = (198, 59, 55)
    else:
        level = 10
        color = (108, 76, 148)
    if is_darker:
        return tuple(array(color) * 2 // 3)
    if is_level:
        return level
    return color


def get_jacket(mid: str, m_type: str, is_small: bool = False) -> str:
    mid = mid.zfill(4)
    target_jk = ('jk_%s_%d.png' % (mid, (int(m_type) + 1)))
    backup_jk = ('jk_%s_1.png' % mid)
    dummy_jk_path = game_dir + '/data/graphics/jk_dummy.png'
    if is_small:
        target_jk = ('jk_%s_%d_s.png' % (mid, (int(m_type) + 1)))
        backup_jk = ('jk_%s_1_s.png' % mid)
        dummy_jk_path = game_dir + '/data/graphics/jk_dummy_s.png'
    for song_folder in listdir(song_folders):
        if song_folder.startswith(mid):
            song_path = ('%s/%s/' % (song_folders, song_folder))
            if path.exists(song_path + target_jk):
                return song_path + target_jk
            elif path.exists(song_path + backup_jk):
                return song_path + backup_jk
    return dummy_jk_path


def get_diff(m_type: str, inf_ver: str) -> str:
    diff_table = [['NOV', 'ADV', 'EXH', '', 'MXM'] for _ in range(4)]
    diff_table[0][3], diff_table[1][3], diff_table[2][3], diff_table[3][3] = 'INF', 'GRV', 'HVN', 'VVD'
    try:
        return diff_table[int(inf_ver) - 2][int(m_type)]
    except IndexError:
        return 'UNK'


def length_uni(font: ImageFont.truetype, text: str, length: int) -> str:
    new_text = ''
    for char in text:
        if font.getsize(new_text)[0] >= length:
            return new_text
        new_text += char
    return new_text


def simple_plot(bg: array, img: array, pos: list, relative: tuple = (0, 0)):
    # pos and relative should be like (y, x)
    img_y, img_x, chn = img.shape
    y, x = pos[0] + relative[0], pos[1] + relative[1]
    bg[y:y + img_y, x:x + img_x] = img


def png_superimpose(bg: array, img: 'PNG image with 4 channels', pos: list, is_trans: bool = False):
    # pos should be like (y, x)
    img_size = img.shape
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            alpha = img[y][x][3] / 255
            if is_trans:
                alpha = 0.5 * alpha
            bg[pos[0] + y][pos[1] + x] = (1 - alpha) * bg[pos[0] + y][pos[1] + x] + alpha * img[y][x][0:3]


def get_ap_card(ap_card: str) -> str:
    card_file = 'ap_'
    card_ver, is_r, is_s = int(ap_card[0]) + 1, (ap_card[1] == '5'), (ap_card[1] == '9')
    if card_ver == 1:
        pass
    else:
        card_file += ('0%s_' % card_ver)
    if is_r:
        card_file += ('R%s' % ap_card[1:].zfill(4))
    elif is_s:
        card_file += ('S%s' % ap_card[1:].zfill(4))
    else:
        card_file += ap_card[1:].zfill(4)
    return game_dir + '/graphics/ap_card/%s.png' % card_file


def plot_single(record: list, profile: list):
    """
    Plot function for single record
    :param record:  a list of [mid, m_type, score, clear, grade, m_time, name, lv, inf_ver, vf]
                    all of them are strings, except vf is a float
    :param profile: a list of [user_name, aka_name, ap_card, skill], all strings
    :return:        image stored as numpy array
    """
    mid, m_type, score, clear, grade, m_time, name, lv, inf_ver, vf = record
    user_name, aka_name, ap_card, skill = profile
    diff = get_diff(m_type, inf_ver)
    real_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(m_time) / 1000))

    bg = cv2.imread(img_archive + 'bg/bg_template.png')

    # Plot appeal card
    ap_card_path = get_ap_card(ap_card)
    card_img = cv2.imread(ap_card_path, cv2.IMREAD_UNCHANGED)
    card_img = cv2.resize(card_img, dsize=None, fx=0.48, fy=0.48, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, card_img, [22, 111])

    # Plot skill
    skill_img = cv2.imread(img_archive + 'skill/skill_' + skill.zfill(2) + '.png', cv2.IMREAD_UNCHANGED)
    skill_img = cv2.resize(skill_img, dsize=None, fx=0.44, fy=0.44, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, skill_img, [79, 203])

    # Plot jacket
    jk_path = get_jacket(mid, m_type)
    jk = cv2.imread(jk_path)
    bg[155:455, 126:426] = jk

    # Get bpm string
    bpm_h, bpm_l = level_table[int(mid)][3], level_table[int(mid)][4]
    if bpm_h[-2:] == '00':
        bpm_h = int(int(bpm_h) / 100)
    else:
        bpm_h = int(bpm_h) / 100
    if bpm_l[-2:] == '00':
        bpm_l = int(int(bpm_l) / 100)
    else:
        bpm_l = int(bpm_l) / 100
    if bpm_h == bpm_l:
        bpm = str(bpm_h)
    else:
        bpm = str(bpm_l) + "~" + str(bpm_h)

    # Plot level box
    level_box = cv2.imread(img_archive + 'level/level_small_' + diff.lower() + '.png', cv2.IMREAD_UNCHANGED)
    level_box = cv2.resize(level_box, dsize=None, fx=0.824, fy=0.824, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, level_box, [474, 352])

    # Get artist string
    artist = level_table[int(mid)][2]

    # Plot clear mark
    mark = cv2.imread(img_archive + 'mark/mark_' + clear_img[clear] + '.png', cv2.IMREAD_UNCHANGED)
    mark = cv2.resize(mark, dsize=None, fx=0.941, fy=0.941, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, mark, [517, 418])

    # Get effector and illustrator string
    ill, eff = level_table[int(mid)][int(m_type) * 3 + 8], level_table[int(mid)][int(m_type) * 3 + 9]

    # Plot score
    score_x, is_zero = 2, True
    score = score.zfill(8)
    h_score, l_score = score[:4], score[4:8]

    for num in h_score:
        score_x += (58 + 1)
        if num != '0':
            is_zero = False
        num_img = cv2.imread(img_archive + 'number/num_score_' + num + '.png', cv2.IMREAD_UNCHANGED)
        png_superimpose(bg, num_img, [690, score_x], is_zero)

    score_x += 56
    for num in l_score:
        if num != '0':
            is_zero = False
        num_img = cv2.imread(img_archive + 'number/num_mmscore_' + num + '.png', cv2.IMREAD_UNCHANGED)
        png_superimpose(bg, num_img, [698, score_x], is_zero)
        score_x += 50

    # Plot grade
    grade_bg = cv2.imread(img_archive + 'grade/box_medal.png', cv2.IMREAD_UNCHANGED)
    grade_bg = cv2.resize(grade_bg, dsize=None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, grade_bg, [775, 94])

    grade_png = cv2.imread(img_archive + 'grade/grade_' + grade_img[grade] + '.png', cv2.IMREAD_UNCHANGED)
    grade_png = cv2.resize(grade_png, dsize=None, fx=0.41, fy=0.41, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, grade_png, [783, 104])

    # Plot all characters
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(bg)
    pen = ImageDraw.Draw(pil_img)

    name_font = ImageFont.truetype(font_DFHS, 34, encoding='utf-8', index=1)
    pen.text((202, 30), user_name, text_white, font=name_font)

    bpm_font = ImageFont.truetype(font_unipace, 16, encoding='utf-8')
    pen.text((166, 473), bpm, text_white, font=bpm_font)

    name_font = ImageFont.truetype(font_DFHS, 22, encoding='utf-8', index=1)
    name_uni = length_uni(name_font, name, 335)
    artist_uni = length_uni(name_font, artist, 335)
    pen.text((69, 517), name_uni, text_white, font=name_font)
    pen.text((69, 560), artist_uni, text_white, font=name_font)

    eff_font = ImageFont.truetype(font_DFHS, 15, encoding='utf-8', index=1)
    eff_uni = length_uni(eff_font, eff, 250)
    ill_uni = length_uni(eff_font, ill, 250)
    pen.text((222, 612), eff_uni, text_white, font=eff_font)
    pen.text((222, 648), ill_uni, text_white, font=eff_font)

    lv_font = ImageFont.truetype(font_unipace, 12, encoding='utf-8')
    pen.text((418, 478), lv, text_white, font=lv_font)

    time_font = ImageFont.truetype(font_DFHS, 22, encoding='utf-8', index=1)
    pen.text((216, 777), 'VOLFORCE', text_white, font=time_font)
    pen.text((216, 815), real_time, text_white, font=time_font)

    vf_font = ImageFont.truetype(font_unipace, 20, encoding='utf-8')
    pen.text((369, 776), '%.3f' % (vf / 2), get_vf_property(vf / 2), font=vf_font)

    bg = cv2.cvtColor(array(pil_img), cv2.COLOR_RGB2BGR)

    # cv2.imshow('test', bg)
    # cv2.waitKey(0)

    # Get recent text message
    msg = ''
    msg += ('Played at %s\n' % real_time)
    msg += ('%s%-2s  %s\n' % (diff, lv, name))
    msg += ('%-9s%-6s%-5s\n' % (score, grade_table[grade], clear_table[clear]))
    msg += ('VF:%.3f\n' % vf)

    print(msg)

    cv2.imwrite('%s/%s_Recent.png' % (output, user_name), bg, params=[cv2.IMWRITE_PNG_COMPRESSION, 3])

    print('Plot successfully.')


def plot_b50(music_b50: list, profile: list):
    """
    Plot function for best 50 records
    :param music_b50: a list of highest vf songs, each line of music_b50 should be:
                      [is_recorded, mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf]
                      all of them are strings, except vf is a float
    :param profile:   a list of [card_num, user_name, aka_name, ap_card, skill], all strings
    :return:          image stored as numpy array
    """
    card_num, user_name, aka_name, ap_card, skill = profile

    # Get overall volforce
    vol_force = 0.0
    for record in music_b50:
        if record[0]:
            vol_force += int(record[-1] * 10) / 1000
        else:
            break

    # Load image files
    bg = cv2.imread(img_archive + 'bg/B50_bg.png')

    level_nov = cv2.imread(img_archive + 'level/level_small_nov.png')
    level_adv = cv2.imread(img_archive + 'level/level_small_adv.png')
    level_exh = cv2.imread(img_archive + 'level/level_small_exh.png')
    level_inf = cv2.imread(img_archive + 'level/level_small_inf.png')
    level_grv = cv2.imread(img_archive + 'level/level_small_grv.png')
    level_hvn = cv2.imread(img_archive + 'level/level_small_hvn.png')
    level_vvd = cv2.imread(img_archive + 'level/level_small_vvd.png')
    level_mxm = cv2.imread(img_archive + 'level/level_small_mxm.png')
    level_list = [level_nov, level_adv, level_exh, '', level_mxm, level_inf, level_grv, level_hvn, level_vvd]

    mark_refactor = 0.58
    mark_cr = cv2.imread(img_archive + 'mark/bg_mark_crash.png')
    mark_cr = cv2.resize(mark_cr, dsize=None, fx=mark_refactor, fy=mark_refactor, interpolation=cv2.INTER_AREA)
    mark_nc = cv2.imread(img_archive + 'mark/bg_mark_comp.png')
    mark_nc = cv2.resize(mark_nc, dsize=None, fx=mark_refactor, fy=mark_refactor, interpolation=cv2.INTER_AREA)
    mark_hc = cv2.imread(img_archive + 'mark/bg_mark_comp_ex.png')
    mark_hc = cv2.resize(mark_hc, dsize=None, fx=mark_refactor, fy=mark_refactor, interpolation=cv2.INTER_AREA)
    mark_uc = cv2.imread(img_archive + 'mark/bg_mark_uc.png')
    mark_uc = cv2.resize(mark_uc, dsize=None, fx=mark_refactor, fy=mark_refactor, interpolation=cv2.INTER_AREA)
    mark_puc = cv2.imread(img_archive + 'mark/bg_mark_puc.png')
    mark_puc = cv2.resize(mark_puc, dsize=None, fx=mark_refactor, fy=mark_refactor, interpolation=cv2.INTER_AREA)
    mark_list = ['', mark_cr, mark_nc, mark_hc, mark_uc, mark_puc]

    grade_refactor = 0.42
    grade_a = cv2.imread(img_archive + 'grade/bg_grade_a.png')
    grade_a = cv2.resize(grade_a, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_ap = cv2.imread(img_archive + 'grade/bg_grade_a_plus.png')
    grade_ap = cv2.resize(grade_ap, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_aa = cv2.imread(img_archive + 'grade/bg_grade_aa.png')
    grade_aa = cv2.resize(grade_aa, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_aap = cv2.imread(img_archive + 'grade/bg_grade_aa_plus.png')
    grade_aap = cv2.resize(grade_aap, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_aaa = cv2.imread(img_archive + 'grade/bg_grade_aaa.png')
    grade_aaa = cv2.resize(grade_aaa, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_aaap = cv2.imread(img_archive + 'grade/bg_grade_aaa_plus.png')
    grade_aaap = cv2.resize(grade_aaap, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_b = cv2.imread(img_archive + 'grade/bg_grade_b.png')
    grade_b = cv2.resize(grade_b, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_c = cv2.imread(img_archive + 'grade/bg_grade_c.png')
    grade_c = cv2.resize(grade_c, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_d = cv2.imread(img_archive + 'grade/bg_grade_d.png')
    grade_d = cv2.resize(grade_d, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_s = cv2.imread(img_archive + 'grade/bg_grade_s.png')
    grade_s = cv2.resize(grade_s, dsize=None, fx=grade_refactor, fy=grade_refactor, interpolation=cv2.INTER_AREA)
    grade_list = ['', grade_d, grade_c, grade_b, grade_a, grade_ap, grade_aa, grade_aap, grade_aaa, grade_aaap, grade_s]

    # Plot images about user profile
    ap_card_path = get_ap_card(ap_card)
    card_img = cv2.imread(ap_card_path, cv2.IMREAD_UNCHANGED)
    card_img = cv2.resize(card_img, dsize=None, fx=0.82, fy=0.82, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, card_img, [44, 827])

    skill_img = cv2.imread(img_archive + 'skill/skill_' + skill.zfill(2) + '.png', cv2.IMREAD_UNCHANGED)
    skill_img = cv2.resize(skill_img, dsize=None, fx=0.54, fy=0.54, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, skill_img, [233, 827])

    vf_level = get_vf_property(vol_force, is_level=True)
    force_img = cv2.imread(img_archive + 'vf/em6_' + str(vf_level).zfill(2) + '_i_eab.png', cv2.IMREAD_UNCHANGED)
    force_img = cv2.resize(force_img, dsize=None, fx=0.34, fy=0.34, interpolation=cv2.INTER_AREA)
    png_superimpose(bg, force_img, [123, 1000])

    # Stipulate relative distance of each elements, due to the OpenCV, the correlate system is (y, x)
    absolute, y_pace, x_pace = (323, 40), 136, 770
    pos_jk, pos_level, pos_mark, pos_grade = (8, 17), (14, 144), (61, 147), (61, 202)

    # Plot image parts
    for index in range(50):
        is_recorded, mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf = music_b50[index]
        if not is_recorded:
            break
        else:
            pass
        base_pos = [absolute[0] + (y_pace * (index // 2)), absolute[1] + (x_pace * (index % 2))]

        # Plot jacket
        jk_path = get_jacket(mid, m_type, True)
        jk = cv2.imread(jk_path)
        simple_plot(bg, jk, base_pos, relative=pos_jk)

        # Plot level box
        if m_type == '3':
            level_box = level_list[int(m_type) + int(inf_ver)]
        else:
            level_box = level_list[int(m_type)]
        simple_plot(bg, level_box, base_pos, relative=pos_level)

        # Plot small icons
        mark_icon = mark_list[int(clear)]
        simple_plot(bg, mark_icon, base_pos, relative=pos_mark)

        grade_icon = grade_list[int(grade)]
        simple_plot(bg, grade_icon, base_pos, relative=pos_grade)

    # Initialize fonts
    user_font = ImageFont.truetype(font_DFHS, 40, encoding='utf-8', index=1)
    id_font = ImageFont.truetype(font_DFHS, 24, encoding='utf-8', index=1)
    vol_font = ImageFont.truetype(font_DFHS, 30, encoding='utf-8', index=1)
    vol_num_font = ImageFont.truetype(font_unipace, 28, encoding='utf-8')
    ceil_font = ImageFont.truetype(font_DFHS, 20, encoding='utf-8', index=1)
    ceil_num_font = ImageFont.truetype(font_unipace, 18, encoding='utf-8')

    level_font = ImageFont.truetype(font_unipace, 15, encoding='utf-8')
    name_font = ImageFont.truetype(font_DFHS, 26, encoding='utf-8', index=1)
    score_h_font = ImageFont.truetype(font_DFHS, 44, encoding='utf-8', index=1)
    score_l_font = ImageFont.truetype(font_DFHS, 32, encoding='utf-8', index=1)
    vf_str_font = ImageFont.truetype(font_DFHS, 32, encoding='utf-8', index=1)
    vf_num_font = ImageFont.truetype(font_unipace, 26, encoding='utf-8')
    rank_font = ImageFont.truetype(font_DFHS, 20, encoding='utf-8', index=1)

    h_size, l_size = score_h_font.getsize('0')[0], score_l_font.getsize('0')[0]

    # Plot all character strings
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(bg)
    pen = ImageDraw.Draw(pil_img)

    # Plot strings of user profile, due to the PIL, the correlate system is (x, y)
    abs_user, abs_id, abs_vol, abs_vol_num, abs_ceil, abs_floor, abs_ceil_num, abs_floor_num = \
        (1020, 55), (1020, 100), (1135, 160), (1344, 161), (1260, 212), (1234, 237), (1344, 214), (1344, 239)
    pos_level_num, pos_name, pos_h_score, pos_vf, pos_vf_num, pos_rank = \
        (221, 19), (271, 14), (271, 63), (510, 73), (570, 75), (700, 93)

    # Check validity of ceil and floor
    if music_b50[0][0]:
        ceil_num = music_b50[0][-1] / 2
    else:
        ceil_num = 0.0
    if music_b50[-1][0]:
        floor_num = music_b50[-1][-1] / 2
    else:
        floor_num = 0.0

    pen.text(abs_user, user_name, text_white, font=user_font)
    pen.text(abs_id, 'ID  ' + card_num, text_white, font=id_font)
    pen.text(abs_vol, 'VOLFORCE', text_white, font=vol_font)
    pen.text(abs_vol_num, ('%.3f' % vol_force), get_vf_property(vol_force), font=vol_num_font)
    pen.text(abs_ceil, 'CEIL', text_white, font=ceil_font)
    pen.text(abs_floor, 'FLOOR', text_white, font=ceil_font)
    pen.text(abs_ceil_num, ('%.3f' % ceil_num), get_vf_property(ceil_num), font=ceil_num_font)
    pen.text(abs_floor_num, ('%.3f' % floor_num), get_vf_property(floor_num), font=ceil_num_font)

    # Plot strings of music records
    def str_plot(text: str, str_font, pos: tuple, color: tuple, length: int = 0, relative: tuple = (0, 0)):
        x, y = pos[0] + relative[0], pos[1] + relative[1]
        if length:
            text = length_uni(str_font, text, length)
        pen.text((x, y), text, color, font=str_font)

    for index in range(50):
        is_recorded, mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf = music_b50[index]
        if not is_recorded:
            break
        base_pos = (absolute[1] + (x_pace * (index % 2)), absolute[0] + (y_pace * (index // 2)))

        # Plot level
        str_plot(lv, level_font, base_pos, text_white, relative=pos_level_num)

        # Plot name
        str_plot(name, name_font, base_pos, text_black, length=450, relative=pos_name)

        # Plot score(both higher bit and lower bit)
        score = score.zfill(8)
        h_score, l_score = score[:4], score[4:8]
        zero_flag = 1
        num_color = text_gray
        x_score = pos_h_score[0]
        for num in h_score:
            if num != '0' and zero_flag:
                num_color = text_black
                zero_flag = 0
            str_plot(num, score_h_font, base_pos, num_color, relative=(x_score, pos_h_score[1]))
            x_score += h_size

        y_l_score = pos_h_score[1] + 10
        for num in l_score:
            if num != '0' and zero_flag:
                num_color = text_black
                zero_flag = 0
            str_plot(num, score_l_font, base_pos, num_color, relative=(x_score, y_l_score))
            x_score += l_size

        # Plot VF("VF" and the vf value)
        str_plot('VF', vf_str_font, base_pos, text_black, relative=pos_vf)
        str_plot(('%.3f' % (vf / 2)), vf_num_font, base_pos, get_vf_property(vf / 2, is_darker=True),
                 relative=pos_vf_num)

        # Plot rank
        str_plot('#%d' % (index + 1), rank_font, base_pos, text_black, relative=pos_rank)

    bg = cv2.cvtColor(array(pil_img), cv2.COLOR_RGB2BGR)

    # Get b50 message
    msg = ''
    msg += ('----------------VOLFORCE %.3f----------------\n' % vol_force)
    msg += 'No.  VF      DIFF   SCORE    RANK  LHT  NAME\n'
    for index in range(50):
        diff = get_diff(music_b50[index][2], music_b50[index][9])
        msg += ('#%-4d%.3f  %s%-2s  %-9s%-6s%-5s%s\n' % ((index + 1), music_b50[index][10], diff,
                                                         music_b50[index][8], music_b50[index][3],
                                                         clear_table[music_b50[index][4]],
                                                         grade_table[music_b50[index][5]], music_b50[index][7]))
    print(msg)

    cv2.imwrite('%s/%s_B50.png' % (output, user_name), bg, params=[cv2.IMWRITE_PNG_COMPRESSION, 3])

    print('Plot successfully.')


def plot_summary(level_summary: array, profile: list):
    """
    Plot function for summarizing the level 17, 18, 19, 20 charts
    :param level_summary: a list of statistics, each line of level summary should be:
                          [NC, HC, UC, PUC, AAA, AAA+, S], all ints
    :param profile:       a list of [card_num, user_name, aka_name, ap_card, skill], all strings
    :return:              image stored as numpy array
    """

    card_num, user_name, aka_name, ap_card, skill = profile

    # Plot user profile images

    # Plot all characters

    # Plot level summary date

    # Get summary message
    msg = ''
    msg += '----------------Level Summary----------------\n' \
           'Level    NC      HC      UC      PUC     ||    AAA     AAA+    S\n'
    for index in range(17, 21):
        la, ls, ld, lf, lg, lh, lj = level_summary[index]
        msg += ('lv.%d    ' % index)
        msg += ('%-8s%-8s%-8s%-8s||    %-8s%-8s%-8s\n' %
                (int(la), int(ls), int(ld), int(lf), int(lg), int(lh), int(lj)))

    print(msg)
    print('Image summary is still on working.')
