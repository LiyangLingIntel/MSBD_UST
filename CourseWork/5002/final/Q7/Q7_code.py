
from itertools import combinations


# initial settings
view_sizes = {
    'abcde': 12,
    'abcd': 9,
    'abce': 10,
    'abde': 8,
    'acde': 7,
    'bcde': 4,
    'abc': 2.5,
    'abd': 3,
    'abe': 5,
    'acd': 2.8,
    'ace': 3,
    'ade': 2.2,
    'bcd': 2,
    'bce': 1.9,
    'bde': 1.7,
    'cde': 3.7,
    'ab': 1.9,
    'ac': 2.1,
    'ad': 0.5,
    'ae': 2.1,
    'bc': 0.7,
    'bd': 0.3,
    'be': 1.4,
    'cd': 0.4,
    'ce': 1.7,
    'de': 0.6,
    'a': 0.1,
    'b': 0.3,
    'c': 0.1,
    'd': 0.2,
    'e': 0.2
}
price_map = dict.fromkeys(view_sizes.keys(), 12)


def subviews(view):
    cb = []
    for i in range(1, len(view)):
        cb.extend([''.join(sorted(c)) for c in combinations(view, i)])
    return cb


def remap_price(view):
    price_map[view] = view_sizes[view]
    for v in subviews(view):
        price_map[v] = view_sizes[view]
    return True


def cal_benefits(view, choices):

    if type(choices) is not list:
        choices = [choices]

    # init with self benefit
    benifits = price_map[view] - view_sizes[view]
    for v in subviews(view):
        if v in choices:
            continue
        discount = price_map[v] - view_sizes[view]
        benifits = benifits + discount if discount > 0 else benifits

    return benifits


def find_views(top_view, n):

    choices = [top_view]
    benefits = []
    while n > 0:
        v_b_tuples = []
        for view in view_sizes.keys():
            if view in choices:
                continue
            cost_reduce = cal_benefits(view, choices)
            v_b_tuples.append((view, cost_reduce))
        best = sorted(v_b_tuples, key=lambda x: x[1], reverse=True)[0]

        remap_price(best[0])
        choices.append(best[0])
        benefits.append(best[1])
        n -= 1

    return choices[1:], benefits


if __name__ == '__main__':

    top_view = 'abcde'
    views, benefits = find_views(top_view, 3)

    for i in range(len(views)):
        print(f'{i+1}th Materialization: {views[i]}, Gain: {benefits[i]}')