#
# WGU C964 Capstone Project
# Equipment Faults in Manufacturing Environments
# Sean Naramor
# July 23, 2021
#
####################################
# This function was used to parse the vast amount of files that contained our data
####################################
def parse_data():
    import os
    import pandas as pd

    path = os.path.dirname(__file__) + '/'
    files = []
    src_path = path+'data_filtered/'
    target_path = path+'data_parsed/'
    file_iterator = os.scandir(src_path)
    # The list of all possible headers. Note that not all files have the same headers.
    # Extra sensors were added to the tool
    # at a point within the selected period of time
    headers = ["ROW_NUMBER", "TIMESTAMP", "Time", "WAFER_ACTIVE", "PPBLOCKNUM", "SYSTEM_ACTIVE", "LOGISTICS", "CONTROLSTATE",
               "PROCESSSTATE", "TC_TEMPERATURE", "PYRO1_TEMPERATURE", "INTENSITY", "I_WARM", "I_COLD", "T_SWITCH", "GAIN",
               "D_GAIN", "SYSTEM_TYPE", "TUBEPYRO", "PORTNUM", "LOTSTATUS", "OXYGEN_SENSOR_(ppm)", "OXYGEN_SENSOR_(%)",
               "Lamp_Current_01", "Lamp_Current_02", "Lamp_Current_03", "Lamp_Current_04", "Lamp_Current_05",
               "Lamp_Current_06", "Lamp_Current_07", "Lamp_Current_08", "Lamp_Current_09", "Lamp_Current_10",
               "Emissivity_1", "Gas1_Flow", "Gas2_Flow", "Gas3_Flow", "Gas4_Flow", "Gas5_Flow"]
    total_data = []
    max_count = 800
    count = 0
    print('=====================================================================================')
    for file in file_iterator:
        if count > max_count:
            break
        with open(os.path.join(src_path, file), 'r', encoding='utf8') as f_open:
            f_timestamp = ''
            f_data_end = 0
            f_data = []
            f_lines = f_open.readlines()
            f_headers = str.split(f_lines[6].rstrip().replace('"', ''), '\t')
            if not headers:
                headers = ['TIMESTAMP']
                headers.extend(f_headers)
            for i, f_line in enumerate(f_lines):
                if f_line.find('START_TIME_FORMAT') >= 0:
                    f_timestamp = f_lines[i + 1].rstrip().split('\t')[1]
                elif f_line.find('NUM_DATA_ROWS') >= 0:
                    f_data_end = i
                    f_stated_length = int(f_lines[i].rstrip().split('\t')[1])
            if f_timestamp and f_data_end:
                raw_data = f_lines[7:f_data_end]
                f_data = [{}] * len(raw_data)

                for j, item in enumerate(raw_data):
                    data_point = {}
                    split = item.rstrip().split('\t')
                    for k, header in enumerate(f_headers):
                        data_point[header] = split[k]

                    data_point["TIMESTAMP"] = f_timestamp
                    data_point["ROW_NUMBER"] = count
                    f_data[j] = data_point
                points_per_run = 10
                mults = []
                # This will create a uniformly spread array of numbers between
                # 0 and 1 (e.g. [0.1, 0.2, 0.3 ... 0.9, 1.0]
                for p in range(points_per_run):
                    val = 1 / points_per_run
                    mults.append((p+1)*val)
                data_length = len(f_data)
                for mult in mults:
                    # Multiply the mult value with the length of data in this file
                    # Round to an integer and you have an index to use
                    # Ex: A file with 100 data points, and we want 3 data points from that file
                    # data point 1 will be line 0, point 2 will be line 50, and point 3 will be line 100
                    index = max(0, min(int(data_length * mult), data_length - 1))
                    if f_data[index]["PROCESSSTATE"] == '5':
                        total_data.append(f_data[index])

                print(f"Data Length: {len(total_data)}")
                # print('=====================================================================================')
        count += 1

    print(f"Total Data Points: {len(total_data)}")

    data_dict = {}
    for header in headers:
        data = []
        for item in total_data:
            if header in item:
                data.append(item[header])
            else:
                data.append('NULL')
        data_dict[header] = data

    df = pd.DataFrame(data_dict)

    df.to_csv(target_path+'data_parsed.csv')
