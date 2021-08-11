import pandas as pd
from datetime import datetime
import datetime as dt
import os, json, gzip, shutil


def split(log_type):
    # log_type can be either 'hash_mapping' or 'country'
    out_folder = log_type
    file = 'hash_mapping.csv' if log_type == 'hash_mapping' else 'combined.csv'

    with open(file, 'r') as infile:
        header = next(infile)
        curr_slug = ''
        for line in infile:
            tokens = line.split(',')
            slug = '{}-{}'.format(tokens[0], tokens[1])
            if curr_slug == slug:
                # same course
                with open(os.path.join(out_folder, '{}.csv'.format(slug)), 'a') as outfile:
                    outfile.write(line)
            else:
                # start of new course
                with open(os.path.join(out_folder, '{}.csv'.format(slug)), 'w+') as outfile:
                    outfile.write(header)
                    outfile.write(line)
                curr_slug = slug


def load_increment_dates():
    print('loading increment dates at {}'.format(datetime.now()))
    df = pd.read_csv('increment_dates.csv')

    increment_dates = dict()
    for index, row in df.iterrows():
        slug = row['course']
        increment_dates[slug] = list()

        end_date = datetime.strptime(row['end_date'], '%m/%d/%y')
        eod = dt.timedelta(hours=23, minutes=59, seconds=59)
        end_date = end_date + eod

        for i in range(1, 9):
            increment_dates[slug].append(datetime.strptime(row['increment_{}_start'.format(i)], '%m/%d/%Y %H:%M:%S'))
        increment_dates[slug].append(end_date)

    return increment_dates


def get_curr_increment(increments, ts):
    increment = -1
    for inc in range(0, 8):
        if increments[inc] <= ts < increments[inc + 1]:
            increment = inc
            break
    return increment


def pull_forum_features(increment_dates):
    '''
    forum posts
    post_id, thread_id, user_id, timestamp

    comments
    comment_id, thread_id, post_id, user_id, timestamp
    '''

    print('starting pulling of incremental forum features at {}'.format(datetime.now()))
    forum_folder = 'forum_data/'
    # forum_folder = 'forum_test/'

    courses = os.listdir(forum_folder)
    for c in courses:
        sessions = os.listdir(os.path.join(forum_folder, c, 'f'))

        for s in sessions:
            threads = dict()
            posts = dict()
            users = dict()

            slug = '{}-{}'.format(c, s.split('-')[1].split('.')[0])
            try:
                increments = increment_dates[slug]
            except KeyError:
                slug = slug.replace('001', '2012-001')
                increments = increment_dates[slug]

            print('pulling forum features from {} at {}'.format(slug, datetime.now()))

            # FORUM POSTS
            with open(os.path.join(forum_folder, c, 'f', s), 'r', encoding='utf8') as infile:
                for line in infile:
                    tokens = line.split('\t')
                    post_id = tokens[0]
                    thread_id = tokens[1]
                    user_id = tokens[2]
                    ts = datetime.fromtimestamp(int(tokens[3])).strftime('%m/%d/%Y %H:%M:%S')
                    ts = datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
                    inc = get_curr_increment(increments, ts)  # starts at 0 (i.e., increment 1 has index 0)

                    if user_id not in users:
                        users[user_id] = [0 for x in range(32)]
                        '''
                        0: num_posts
                        1: num_started
                        2: num_responses
                        3: num_respondents
                        '''
                    if post_id not in posts:
                        posts[post_id] = user_id

                    if inc >= 0:
                        index = 0 + (4 * inc)
                        users[user_id][index] += 1

                    if thread_id not in threads:
                        threads[thread_id] = user_id

                        if inc >= 0:
                            users[user_id][index + 1] += 1
                    else:
                        if inc >= 0:
                            users[user_id][index + 2] += 1
                            users[threads[thread_id]][index + 3] += 1

            # COMMENTS
            with open(os.path.join(forum_folder, c, 'c', s.replace('f', 'c')), 'r', encoding='utf8') as infile:
                for line in infile:
                    tokens = line.split('\t')
                    post_id = tokens[2]
                    user_id = tokens[3]
                    ts = datetime.fromtimestamp(int(tokens[4])).strftime('%m/%d/%Y %H:%M:%S')
                    ts = datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
                    inc = get_curr_increment(increments, ts)  # starts at 0 (i.e., increment 1 has index 0

                    if inc >= 0:
                        if user_id not in users:
                            users[user_id] = [0 for x in range(32)]
                        '''
                        0: num_posts
                        1: num_started
                        2: num_responses
                        3: num_respondents
                        '''

                        index = 0 + (4 * inc)
                        users[user_id][index] += 1
                        users[user_id][index + 2] += 1
                        users[posts[post_id]][index + 3] += 1

            with open('forum_features-output/{}.csv'.format(slug), 'w+') as outfile:
                outfile.write('slug,user_id,num_posts_inc1,num_started_inc1,num_responses_inc1,num_respondents_inc1,'
                              'num_posts_inc2,num_started_inc2,num_responses_inc2,num_respondents_inc2,'
                              'num_posts_inc3,num_started_inc3,num_responses_inc3,num_respondents_inc3,'
                              'num_posts_inc4,num_started_inc4,num_responses_inc4,num_respondents_inc4,'
                              'num_posts_inc5,num_started_inc5,num_responses_inc5,num_respondents_inc5,'
                              'num_posts_inc6,num_started_inc6,num_responses_inc6,num_respondents_inc6,'
                              'num_posts_inc7,num_started_inc7,num_responses_inc7,num_respondents_inc7,'
                              'num_posts_inc8,num_started_inc8,num_responses_inc8,num_respondents_inc8\n')
                for user_id in users:
                    outfile.write('{},{}'.format(slug, user_id))
                    for i in range(32):
                        outfile.write(',{}'.format(users[user_id][i]))
                    outfile.write('\n')


def pull_incremental_features(increment_dates, return_days_active=False):
    print('starting pulling of incremental features at {}'.format(datetime.now()))
    click_folder = 'clickstream_data/'
    # click_folder = 'test/'
    courses = os.listdir(click_folder)

    for c in courses:
        sessions = os.listdir(os.path.join(click_folder, c))
        for s in sessions:
            slug = '{}-{}'.format(c, s)
            increments = increment_dates[slug]
            usernames = list()

            forum_views = dict()
            quiz_views = dict()
            peer_views = dict()
            lecture_views = dict()
            days_active = dict()

            gz_clickstream = os.listdir(os.path.join(click_folder, c, s))[0]
            clickstream = gz_clickstream.replace('.gz', '')

            print('gunzipping {} at {}'.format(slug, datetime.now()))
            with gzip.open(os.path.join(click_folder, c, s, gz_clickstream), 'rb') as f_in:
                with open(os.path.join(click_folder, c, s, clickstream), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print('pulling features from {} at {}'.format(slug, datetime.now()))
            with open(os.path.join(click_folder, c, s, clickstream), 'r', encoding='utf8') as infile:
                for line in infile:
                    try:
                        js = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print('{}: ERROR IN {}'.format(datetime.now(), line))
                        continue

                    key = js['key']
                    username = js['username']
                    ts = datetime.fromtimestamp(int(str(js['timestamp'])[:10])).strftime('%m/%d/%Y %H:%M:%S')
                    ts = datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
                    page_url = js['page_url']
                    # print(js)
                    # print(key, username, ts, page_url)
    
                    # DAYS ACTIVE
                    if username not in usernames:
                        usernames.append(username)
                    increment = get_curr_increment(increments, ts)
                    if increment >= 0:
                        if username not in days_active:
                            days_active[username] = [list() for i in range(8)]
                        if ts.date() not in days_active[username][increment]:
                            days_active[username][increment].append(ts.date())
    
                    # INCREMENTAL FEATURES: FORUM, QUIZ, PEER-ASSESSMENT VIEWS
                        if '/forum/' in page_url:
                            if username not in forum_views:
                                forum_views[username] = [0 for i in range(8)]
                            forum_views[username][increment] += 1
                        elif '/quiz/' in page_url:
                            if username not in quiz_views:
                                quiz_views[username] = [0 for i in range(8)]
                            quiz_views[username][increment] += 1
                        elif key == 'hg.hg.pageview' or key == 'hg.pageview':
                            if username not in peer_views:
                                peer_views[username] = [0 for i in range(8)]
                            peer_views[username][increment] += 1
                        elif '/lecture/' in page_url:
                            if username not in lecture_views:
                                lecture_views[username] = [0 for i in range(8)]
                            lecture_views[username][increment] += 1
    
            print('printing output at {}'.format(datetime.now()))
            with open('output/{}-incremental_features.csv'.format(slug), 'w+') as outfile:
                outfile.write('course_slug,username,forum_views_inc1,forum_views_inc2,forum_views_inc3,'
                              'forum_views_inc4,forum_views_inc5,forum_views_inc6,forum_views_inc7,forum_views_inc8,'
                              'quiz_views_inc1,quiz_views_inc2,quiz_views_inc3,quiz_views_inc4,quiz_views_inc5,'
                              'quiz_views_inc6,quiz_views_inc7,quiz_views_inc8,peer_views_inc1,peer_views_inc2,'
                              'peer_views_inc3,peer_views_inc4,peer_views_inc5,peer_views_inc6,peer_views_inc7,'
                              'peer_views_inc8,lect_views_inc1,lect_views_inc2,lect_views_inc3,lect_views_inc4,'
                              'lect_views_inc5,lect_views_inc6,lect_views_inc7,lect_views_inc8,days_active_inc1,'
                              'days_active_inc2,days_active_inc3,days_active_inc4,days_active_inc5,days_active_inc6,'
                              'days_active_inc7,days_active_inc8\n')
                for username in usernames:
                    outfile.write('{},{}'.format(slug, username))
    
                    for i in range(0, 8):
                        try:
                            outfile.write(',{}'.format(forum_views[username][i]))
                        except KeyError:
                            outfile.write(',0')
                    for i in range(0, 8):
                        try:
                            outfile.write(',{}'.format(quiz_views[username][i]))
                        except KeyError:
                            outfile.write(',0')
                    for i in range(0, 8):
                        try:
                            outfile.write(',{}'.format(peer_views[username][i]))
                        except KeyError:
                            outfile.write(',0')
                    for i in range(0, 8):
                        try:
                            outfile.write(',{}'.format(lecture_views[username][i]))
                        except KeyError:
                            outfile.write(',0')
                    for i in range(0, 8):
                        try:
                            outfile.write(',{}'.format(len(days_active[username][i])))
                        except KeyError:
                            outfile.write(',0')
                    outfile.write('\n')

            # print(type(click_folder), type(c), type(s), type(clickstream))
            os.remove(os.path.join(click_folder, c, s, clickstream))


def pull_durations(increment_dates):
    print('pulling time spent features at {}'.format(datetime.now()))
    click_folder = 'clickstream_data'

    courses = os.listdir(click_folder)
    for c in courses:
        sessions = os.listdir(os.path.join(click_folder, c))
        for s in sessions:
            slug = '{}-{}'.format(c, s)
            increments = increment_dates[slug]
            usernames = dict()

            gz_clickstream = os.listdir(os.path.join(click_folder, c, s))[0]
            clickstream = gz_clickstream.replace('.gz', '')

            print('gunzipping {} at {}'.format(slug, datetime.now()))
            with gzip.open(os.path.join(click_folder, c, s, gz_clickstream), 'rb') as f_in:
                with open(os.path.join(click_folder, c, s, clickstream), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print('pulling time-spent features from {} at {}'.format(slug, datetime.now()))
            with open(os.path.join(click_folder, c, s, clickstream), 'r', encoding='utf8') as infile:
                for line in infile:
                    try:
                        line = line.replace('\\\'', '')
                        js = json.loads(line)
                    except json.decoder.JSONDecodeError:
                        print('{}: ERROR IN {}'.format(datetime.now(), line))
                        continue

                    key = js['key']
                    username = js['username']
                    ts = datetime.fromtimestamp(int(str(js['timestamp'])[:10])).strftime('%m/%d/%Y %H:%M:%S')
                    ts = datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
                    page_url = js['page_url']

                    if '/forum/' in page_url:
                        page = 'forum'
                    elif '/quiz/' in page_url:
                        page = 'quiz'
                    elif key == 'hg.hg.pageview' or key == 'hg.pageview':
                        page = 'peer'
                    elif '/lecture/' in page_url:
                        page = 'lecture'
                    else:
                        page = '.'

                    if username not in usernames:
                        usernames[username] = pd.DataFrame(columns=['timestamp', 'page'])

                    usernames[username] = usernames[username].append({'timestamp': ts, 'page': page}, ignore_index=True)

            os.remove(os.path.join(click_folder, c, s, clickstream))

            print('printing {} to output'.format(slug))
            with open('durations/{}-durations.csv'.format(slug), 'w+') as outfile:
                outfile.write('slug,username,ts,duration_sec,page\n')
                for username in usernames:
                    df = usernames[username]
                    df = df.sort_values(by=['timestamp'])
                    df['duration'] = compute_durations(username, df)
                    for index, row in df.iterrows():
                        outfile.write('{},{},{},{},{}\n'.format(
                            slug, username, row['timestamp'], row['duration'], row['page']))


def compute_durations(username, df):
    # print('computing durations for user {} at {}'.format(username, datetime.now()))
    durations = list()

    for i in range(len(df) - 1):
        curr_time = df.iloc[i, df.columns.get_loc('timestamp')]
        next_time = df.iloc[i + 1, df.columns.get_loc('timestamp')]
        duration = next_time - curr_time

        if duration >= dt.timedelta(hours=1):
            durations.append('.')
        else:
            durations.append(duration.total_seconds())
    durations.append('.')

    return durations


def pull_time_spent_features(increment_dates):
    print('pulling time spent features at', datetime.now())
    durations = 'durations/'

    for f in os.listdir(durations):
        slug = f[:-14]
        increments = increment_dates[slug]
        print('pulling data from {} at {}'.format(slug, datetime.now()))

        usernames = list()
        forum_times = dict()
        lecture_times = dict()
        quiz_times = dict()
        peer_times = dict()

        df = pd.read_csv(os.path.join(durations, f))
        for index, row in df.iterrows():
            ts = datetime.strptime(row['ts'], '%Y-%m-%d %H:%M:%S')
            increment = get_curr_increment(increments, ts)
            if increment < 0:
                # click happened before or after the official run of the course
                continue

            try:
                duration = float(row['duration_sec'])
            except ValueError:
                # click has no recorded duration (e.g., final learner's action, exceeded more than an hour (disengaged))
                continue

            page = row['page']
            if page == '.':
                # click is neither in lecture, forum, peer, or quiz
                continue
            else:
                username = row['username']
                if username not in usernames:
                    usernames.append(username)
                    forum_times[username] = [0 for i in range(8)]
                    lecture_times[username] = [0 for i in range(8)]
                    peer_times[username] = [0 for i in range(8)]
                    quiz_times[username] = [0 for i in range(8)]

                if page == 'forum':
                    forum_times[username][increment] += duration
                elif page == 'lecture':
                    lecture_times[username][increment] += duration
                elif page == 'peer':
                    peer_times[username][increment] += duration
                elif page == 'quiz':
                    quiz_times[username][increment] += duration

        with open('time_spent-output/{}.csv'.format(slug), 'w+') as outfile:
            outfile.write('username,lect_inc1,lect_inc2,lect_inc3,lect_inc4,lect_inc5,lect_inc6,lect_inc7,lect_inc8,'
                          'forum_inc1,forum_inc2,forum_inc3,forum_inc4,forum_inc5,forum_inc6,forum_inc7,forum_inc8,'
                          'peer_inc1,peer_inc2,peer_inc3,peer_inc4,peer_inc5,peer_inc6,peer_inc7,peer_inc8,'
                          'quiz_inc1,quiz_inc2,quiz_inc3,quiz_inc4,quiz_inc5,quiz_inc6,quiz_inc7,quiz_inc8\n')
            for username in usernames:
                outfile.write(username)
                for i in range(8):
                    outfile.write(',{}'.format(lecture_times[username][i]))
                    outfile.write(',{}'.format(forum_times[username][i]))
                    outfile.write(',{}'.format(peer_times[username][i]))
                    outfile.write(',{}'.format(quiz_times[username][i]))
                outfile.write('\n')


def merge_data_streams():
    mapping_folder = 'hash_mapping/'
    country_folder = 'country/'
    grades_folder = 'course_grades/'
    inter_folder = 'interaction_features-output/'
    time_spent_folder = 'time_spent-output/'
    forum_folder = 'forum_features-output/'
    output_folder = 'merged/'

    for f in os.listdir(mapping_folder):
        slug = f[:-4]
        print('starting to merge {} at {}'.format(slug, datetime.now()))

        id_map = dict()

        country_drop = ['drop_percent', 'z_score_forum_views', 'z_score_days_active', 'z_score_quiz_views',
                        'z_score_exam_views', 'z_score_peer_assessment_views', 'num_posts', 'num_responses',
                        'num_respondents', 'course', 'session']
        country_df = pd.read_csv(os.path.join(country_folder, f), keep_default_na=False).drop(country_drop, axis=1)
        country_df.rename(columns={'user_id': 'session_user_id'}, inplace=True)

        map_df = pd.read_csv(os.path.join(mapping_folder, f))
        # for index, row in map_df.iterrows():
        #     id_map[row['user_id']] = row['session_user_id']

        grades_df = pd.read_csv(os.path.join(grades_folder, f.replace('.csv', '.txt'))).drop(['user_id'], axis=1)

        inter_df = pd.read_csv(os.path.join(inter_folder, f.replace('.csv', '-incremental_features.csv'))).drop(['course_slug'], axis=1)
        inter_df.rename(columns={'username': 'session_user_id'}, inplace=True)

        # time_spent features, username (i.e., session_user_id)
        ts_df = pd.read_csv(os.path.join(time_spent_folder, f))
        ts_df.rename(columns={'username': 'session_user_id'}, inplace=True)

        forum_df = pd.read_csv(os.path.join(forum_folder, f)).drop(['slug'], axis=1)

        print('dropping because not in grades: {}'.format(len(map_df.index) - len(grades_df.index)))
        merged_df = map_df.merge(grades_df, on='session_user_id', how='inner')

        print('dropping because not in clickstream: {}'.format(len(merged_df.index) - len(country_df.index)))
        merged_df = merged_df.merge(country_df, on='session_user_id', how='inner')

        merged_df = merged_df.merge(inter_df, on='session_user_id', how='left')
        merged_df = merged_df.merge(ts_df, on='session_user_id', how='left')
        merged_df = merged_df.merge(forum_df, on='user_id', how='left')
        merged_df.fillna(0).to_csv(os.path.join(output_folder, f), index=False)


def main():
    # increment_dates = load_increment_dates()
    # pull_incremental_features(increment_dates)
    # pull_forum_features(increment_dates)
    # pull_durations(increment_dates)
    # pull_time_spent_features(increment_dates)
    # split(log_type='country')  # options: 'hash_mapping', 'country'
    merge_data_streams()


if __name__ == '__main__':
    main()

