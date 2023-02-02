def fps_change(device):
    '''when fps changed, animate it.'''
    tens_digit = "%d"%int(output.shape[0]//10)
    ones_digit = "%d"%int(output.shape[0]%10)
    def helper(current_y):
        with canvas(device) as draw:
            text(draw, (0, 1), tens_digit, fill="white", font=proportional(CP437_FONT))
            text(draw, (17, current_y), ones_digit, fill="white", font=proportional(CP437_FONT))
        time.sleep(0.1)
    for current_y in range(1, 9):
        helper(current_y)
    ones_digit = "%d"%int(output.shape[0]%10)
    for current_y in range(9, 1, -1):
        helper(current_y)
       
def animation(device, from_y, to_y):
    '''Animate the whole thing, moving it into/out of the abyss.'''
    tens_digit = "%d"%int(output.shape[0]//10)
    ones_digit = "%d"%int(output.shape[0]%10)
    current_y = from_y
    while current_y != to_y:
        with canvas(device) as draw:
            text(draw, (0, current_y), tens_digit, fill="white", font=proportional(CP437_FONT))
            text(draw, (17, current_y), ones_digit, fill="white", font=proportional(CP437_FONT))
        time.sleep(0.1)
        current_y += 1 if to_y > from_y else -1


def main():
    animation(device, 8, 1)
    while True:
        tens_digit = "%d"%int(output.shape[0]//10)
        ones_digit = "%d"%int(output.shape[0]%10)
        with canvas(device) as draw:
            text(draw, (0, 1), tens_digit, fill="white", font=proportional(CP437_FONT))
            text(draw, (17, 1), ones_digit, fill="white", font=proportional(CP437_FONT))
        time.sleep(0.5)
