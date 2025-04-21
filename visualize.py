from quip_api import quip_api, get_quoted_segments

if __name__ == '__main__':
    # text = '''Jeopardy! was created in 1964 by Merv Griffin, while Wheel of Fortune was created in 1975 by Merv Griffin and Roy Leonard. Therefore, Jeopardy! is older than Wheel of Fortune.'''
    text = '''Every player who has won this award and has been eligible for the Naismith Memorial Basketball Hall of Fame has been inducted. Kareem Abdul-Jabbar won the award a record six times. Both Bill Russell and Michael Jordan won the award five times, while Wilt Chamberlain and LeBron James won the award four times. Russell and James are the only players to have won the award four times in five seasons.'''
    quip_res = quip_api([text])[0]

    # naive highlighting
    # rendered_text = ''
    # for is_member, segment in zip(quip_res['is_member'], quip_res['segments']):
    #     if is_member:
    #         rendered_text += '\\textbf{' + segment[0] + '}'
    #     else:
    #         rendered_text += segment[0]
    # # rendered_text += '|'
    # end = '\\textbf{' + segment[1:] + '}' if is_member else segment[1:]
    # rendered_text += end
    # print(rendered_text)
    
    # color gradient based on number of overlapped segments
    mapping = {
        0: lambda x: x,
        1: lambda x: '\quoteda{'+x+'}',
        2: lambda x: '\quotedb{'+x+'}',
        3: lambda x: '\quotedc{'+x+'}',
    }

    postprocessed_text = ''
    for seg in quip_res['segments']:
        postprocessed_text += seg[0]
    postprocessed_text += seg[1:]

    granularity = len(quip_res['segments'][0])
    text_len = len(quip_res['segments'])+granularity-1 # this is not equal to the length of raw text because quip api preprocesses the text
    assert len(postprocessed_text) == text_len

    num_overlap = [0] * text_len
    quoted_segments = get_quoted_segments(quip_res, return_idx=True)
    for start, end in quoted_segments:
        real_end = min(end+granularity-1, text_len-1)
        for i in range(start, real_end+1):
            num_overlap[i] += 1
    # > 3 is treated as 3
    num_overlap = [min(x, 3) for x in num_overlap]
    rendered_text = ''.join([mapping[o](x) for o, x in zip(num_overlap, postprocessed_text)])

    print(rendered_text)
    print(quoted_segments)
    print(quip_res['quip_report']['quip_25_beta'])
    # breakpoint()

