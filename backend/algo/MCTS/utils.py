

def check_requests(requests, t):
    "Checks if there is a new request at time=t."
    if requests[-1].request_time == t:
        return requests[-1]


def get_request_by_id(requests, id):
    "Returns a request."
    for i in range(len(requests)):
        if requests[i].id == id:
            return i


