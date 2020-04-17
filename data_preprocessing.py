import os
import re



def preprocess_frequency_words(language):

    main_directory = 'FrequencyWords/content/2016'

    conversion = {
        'english': 'en',
        'french': 'fr',
        'german': 'de',
        'spanish': 'es',
        'swedish': 'sv',
        'italian': 'it',
        'norwegian': 'no',
        'russian': 'ru',
        'romanian': 'ro'
    }


    converted_language = conversion[language]

    input_file = os.path.join(main_directory, converted_language, f'{converted_language}_50k.txt')
    output_file =os.path.join('/Users/Max/Documents/AI/languagePredictor/clean_data/full', f'{language}.csv')

    with open(output_file, 'w') as out_f:
        with open(input_file, 'r') as in_f:
            i = 0
            for line in in_f:
                # print('x')
                l = line.strip().split(' ')[0]
                # print(l)
                m = re.match('\A[a-zA-Z\u00C0-\u00FF]*$', l)
                # accent_match = re.match('[\u00C0-\u00FF]*', l)
                # print(m)
                # if accent_match:
                #     print(l)
                if m:
                    # print(m)
                    out_f.write(l + '\n')
                    i += 1

            print(i)

def preprocess_sketch_engine(language):
    files = [file for file in os.listdir('./unclean_data') if language in file]
    print(files)

    with open(f'clean_data/{language}.csv', 'w') as out_csv:
        for file in files:
            with open(file, 'r') as in_csv:

                for i, line in enumerate(in_csv):
                    print('x')
                    if i > 3:
                        print(line)
                        # word = line.split()[1]
                        # out_csv.write(word + '\n')


def main():
    languages = [
                    'english',
                    'french',
                    'german',
                    'spanish',
                    'swedish',
                    'italian',
                    'norwegian',
                    'russian',
                    'romanian'
                    # 'italian',
                    # 'russian'
    ]
    # preprocess_english()
    for language in languages:
        # print(language)
        preprocess_frequency_words(language)


if __name__ == '__main__':
    main()
