from src.utils.special_print import print_highlighted


def get_port():
    for i in range(3000, 4000):
        if check_port(i):
            print_highlighted("PORT " + str(i) + " IS SELECTED")
            return i
    print("No free ports")
    exit(1)


def check_port(i):
    import socket, errno
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", i))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            print("Port is already in use: ", i)
        else:
            print(e)
        return False
    s.close()
    return True
