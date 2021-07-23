from cfg_read import *
from os import path
from xml.etree.cElementTree import parse
from numpy import array, save, load, zeros

import plot_gen6

clear_factor = {'1': 0.5, '2': 1.0, '3': 1.02, '4': 1.05, '5': 1.10}
clear_index = {2: 0, 3: 1, 4: 2, 5: 3}

grade_factor = {'1': 0.80, '2': 0.82, '3': 0.85, '4': 0.88, '5': 0.91,
                '6': 0.94, '7': 0.97, '8': 1.00, '9': 1.02, '10': 1.05}
grade_index = {8: 4, 9: 5, 10: 6}

get_card = re.compile(r'"__refid":"+[^"]*')
get_name = re.compile(r'"name":"+[^"]*')
get_aka = re.compile(r'"akaname":+[^,]*')
get_ap_card = re.compile(r'"appeal":+[^,]*')
get_base = re.compile(r'"base":+[^,]*')

get_mid = re.compile(r'"mid":+[^,]*')
get_type = re.compile(r'"type":+[^,]*')
get_score = re.compile(r'"score":+[^,]*')
get_clear = re.compile(r'"clear":+[^,]*')
get_grade = re.compile(r'"grade":+[^,]*')
get_time = re.compile(r'"updatedAt":+[^,]*')

basic_msg = '1----Best 50 Songs and VF analysis\n2----Recent play record\n3----Specific song record\n' \
            '4----User summary\n9----Show available skin list\n0----Exit\nEnter corresponding number to continue:'
specific_msg = 'NOV->1, ADV->2, EXH->3, INF/GRV/HVN/VVD/MXM->4\nSearch highest difficulty as default\n' \
               'Enter operators like \'[mid] [diff(optional)]\':'
skin_msg = 'gen6'


def get_skin_type(skin_name: str):
    if skin_name == 'gen6':
        return plot_gen6
    else:
        input('Invalid skin name, please recheck your configurations.')
        sys.exit(1)


class SdvxData:

    def __init__(self):

        # Read config.txt
        self.map_size, self.card_num, self.local_dir, self.db_dir, self.game_dir, self.output, self.skin_name, self.is_init = get_cfg()
        self.plot_skin = get_skin_type(self.skin_name)

        # Validity check
        if not path.exists(self.db_dir):
            input(r'sdvx@asphyxia.db not found, please check your file directory.')
            sys.exit(1)
        if not path.exists(self.game_dir):
            input(r'KFC-**********\contents\data not found, please check your file directory.')
            sys.exit(1)
        if not path.exists(self.output):
            input(r'Output folder not found, please check your file directory.')
            sys.exit(1)

        # level_table.npy check
        if not self.is_init:
            print('Initializing.')

            # Set up music_db encoded with UTF-8
            jis_xml = open(self.game_dir + r'/others/music_db.xml', 'r', encoding='cp932').readlines()
            utf_xml = open(self.local_dir + r'/music_db_utf8.xml', 'w', encoding='utf-8')
            utf_xml.write('<?xml version="1.0" encoding="utf-8"?>\n')
            jis_xml.pop(0)
            for line in jis_xml:
                utf_xml.write(line)
            utf_xml.close()
            music_xml = 'music_db_utf8.xml'

            # Get level information from xml, then saved as npy file
            tree = parse(music_xml)
            root = tree.getroot()
            music_map = [[''] * 22 for _ in range(self.map_size)]
            for index in range(self.map_size):
                try:
                    mid = int(root[index].attrib['id'])
                    name = root[index][0][1].text
                    artist = root[index][0][3].text
                    bpm_max = int(root[index][0][6].text)
                    bpm_min = int(root[index][0][7].text)
                    version = int(root[index][0][13].text)
                    inf_ver = int(root[index][0][15].text)

                    nov_lv = int(root[index][1][0][0].text)
                    nov_ill = root[index][1][0][1].text
                    nov_eff = root[index][1][0][2].text

                    adv_lv = int(root[index][1][1][0].text)
                    adv_ill = root[index][1][1][1].text
                    adv_eff = root[index][1][1][2].text

                    exh_lv = int(root[index][1][2][0].text)
                    exh_ill = root[index][1][2][1].text
                    exh_eff = root[index][1][2][2].text

                    inf_lv = int(root[index][1][3][0].text)
                    inf_ill = root[index][1][3][1].text
                    inf_eff = root[index][1][3][2].text
                    try:
                        mxm_lv = int(root[index][1][4][0].text)
                        mxm_ill = root[index][1][4][1].text
                        mxm_eff = root[index][1][4][2].text
                    except IndexError:
                        mxm_lv = 0
                        mxm_ill = 'dummy'
                        mxm_eff = 'dummy'
                    music_map[int(mid)] = [mid, name, artist, bpm_max, bpm_min, version, inf_ver, nov_lv, nov_ill,
                                           nov_eff, adv_lv, adv_ill, adv_eff, exh_lv, exh_ill, exh_eff, inf_lv,
                                           inf_ill, inf_eff, mxm_lv, mxm_ill, mxm_eff]

                except IndexError:
                    break

            music_map = array(music_map)
            save(self.local_dir + '/level_table.npy', music_map)

            # Add flag to config.txt
            raw_file = open(self.local_dir + '/config.txt', 'a')
            raw_file.write('is initialized=True\n')
            raw_file.close()

            print('Initialization complete.')

        # Read sdvx@asphyxia.db
        self.raw_data = open(self.db_dir, 'r')
        self.raw_music = []
        self.raw_profile = ''
        self.raw_skill = ''

        # Get raw data from db
        for line in self.raw_data:
            if re.search(r'"collection":"music"', line):
                raw_card = get_card.search(line).group()[11:]
                if raw_card == self.card_num:
                    self.raw_music.append(line)
            elif re.search(r'"collection":"profile"', line):
                raw_card = get_card.search(line).group()[11:]
                if raw_card == self.card_num:
                    self.raw_profile = line
            elif re.search(r'"collection":"skill"', line):
                raw_card = get_card.search(line).group()[11:]
                if raw_card == self.card_num:
                    self.raw_skill = line

        if not self.raw_profile:
            input('Card not found, please recheck your card number, or ensuring that you have saved yet.')
            sys.exit(1)
        if not self.raw_music:
            input('Music record not found, please recheck your card number, '
                  'or ensuring that you have played at least once.')
            sys.exit(1)

        # Specify profile data
        self.user_name = get_name.search(self.raw_profile).group()[8:]
        self.aka = get_aka.search(self.raw_profile).group()[10:]
        self.ap_card = get_ap_card.search(self.raw_profile).group()[9:]
        self.skill = get_base.search(self.raw_skill).group()[7:]

        # Specify music data
        self.raw_music.reverse()
        self.level_table = load('level_table.npy')
        self.music_map = [[False, '', '', '', '', '', '', '', '', '', 0.0] for _ in range(self.map_size * 5 + 1)]
        # Each line of music map should be
        # [is_recorded, mid, type, score, clear, grade, timestamp, name, lv, inf_ver, vf]

    def vf_calculator(self, mid: str, m_type: str, score: str, clear: str, grade: str) -> float:
        lv = self.level_table[int(mid)][int(m_type) * 3 + 7]
        try:
            vf = int(lv) * (int(score) / 10000000) * clear_factor[clear] * grade_factor[grade] * 2
        except ValueError:
            return 0.0
        return vf

    def get_music_attr(self, record: str):
        mid = get_mid.search(record).group()[6:]
        m_type = get_type.search(record).group()[7:]
        score = get_score.search(record).group()[8:]
        clear = get_clear.search(record).group()[8:]
        grade = get_grade.search(record).group()[8:]
        m_time = get_time.search(record).group()[22:35]

        lv = self.level_table[int(mid)][int(m_type) * 3 + 7]
        inf_ver, name = self.level_table[int(mid)][6], self.level_table[int(mid)][1]
        if not lv:
            lv, inf_ver = '0', '0'

        return mid, m_type, score, clear, grade, m_time, name, lv, inf_ver

    def data_cleaning(self):
        for record in self.raw_music:
            mid, m_type, score, clear, grade, m_time, name, lv, inf_ver = self.get_music_attr(record)
            music_index = int(mid) * 5 + int(m_type)
            if not self.music_map[music_index][0]:
                vf = self.vf_calculator(mid, m_type, score, clear, grade)
                self.music_map[music_index] = [True, mid, m_type, score, clear, grade, m_time, name, lv, inf_ver, vf]

    def get_b50(self):
        self.data_cleaning()
        self.music_map.sort(key=lambda x: x[10], reverse=True)
        self.plot_skin.plot_b50(self.music_map[0:50], [self.card_num, self.user_name, self.aka, self.ap_card, self.skill])

    def get_recent(self):
        recent = self.raw_music[0]
        mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver = self.get_music_attr(recent)
        vf = self.vf_calculator(mid, m_type, score, clear, grade)
        self.plot_skin.plot_single([mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf],
                            [self.user_name, self.aka, self.ap_card, self.skill])

    def get_specific(self, arg_mid: int, arg_type: int = 5):
        # Get specific chart
        # -1 will be returned if there is no corresponding record
        self.data_cleaning()
        if arg_type == 5:
            index = 0
            for m_index in range(4, -1, -1):
                index = arg_mid * 5 + m_index
                if self.music_map[index][0]:
                    break
                index = 0
            if not index:
                return 1
        else:
            index = arg_mid * 5 + arg_type
            if not self.music_map[index][0]:
                return 2
        is_recorded, mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf = self.music_map[index]
        self.plot_skin.plot_single([mid, m_type, score, clear, grade, timestamp, name, lv, inf_ver, vf],
                             [self.user_name, self.aka, self.ap_card, self.skill])

    def get_summary(self):
        # Dispose summary data
        level_summary = zeros((21, 7))
        # Each line of level summary should be
        # [NC, HC, UC, PUC, AAA, AAA+, S]
        for record in self.music_map:
            if not record[0]:
                break
            clear, grade, lv = int(record[4]), int(record[5]), int(record[11])
            if clear > 1:
                level_summary[lv][clear_index[clear]] += 1
            if grade > 7:
                level_summary[lv][grade_index[grade]] += 1

        self.plot_skin.plot_summary(level_summary, [self.card_num, self.user_name, self.aka, self.ap_card, self.skill])


if __name__ == '__main__':
    base = SdvxData()

    while True:
        op_num = input(basic_msg)
        print()

        if op_num == '1':
            base.get_b50()
        elif op_num == '2':
            base.get_recent()
        elif op_num == '3':
            op_spe = input(specific_msg).split()
            if len(op_spe) == 1:
                if base.get_specific(int(op_spe[0])) == 1:
                    input('Invalid operator.')
                elif base.get_specific(int(op_spe[0])) == 2:
                    input('Record not found.')
            elif len(op_spe) == 2:
                op_mid, op_type = int(op_spe[0]), int(op_spe[1])
                if base.get_specific(op_mid, op_type) == 1:
                    input('Invalid operator.')
                elif base.get_specific(op_mid, op_type) == 2:
                    input('Record not found.')
            else:
                input('Invalid operator.')
        elif op_num == '4':
            base.get_summary()
        elif op_num == '9':
            input(skin_msg)
        else:
            break

        input('Press enter to continue.')
