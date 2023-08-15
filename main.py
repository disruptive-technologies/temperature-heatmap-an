import argparse

import disruptive as dt

from heatmap.heatmap import Heatmap

# SA_KEY_ID: str = '<SERVICE_ACCOUNT_KEY>'
# SA_SECRET: str = '<SERVICE_ACCOUT_SECRET>'
# SA_EMAIL: str = '<SERVICE_ACCOUNT_EMAIL>'


def parse_sysargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # The following arguments are optional.
    parser.add_argument(
        '--layout',
        type=str,
        default='',
        help='Target layout JSON file.',
    )

    # The following arguments are boolean flags.
    parser.add_argument(
        '--threaded',
        action='store_true',
        help='Run heatmap generation on multiple threads.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Plot heatmap as it is constructed.',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_sysargs()

    # Authenticate Disruptive Python Client.
    # dt.default_auth = dt.Auth.service_account(SA_KEY_ID, SA_SECRET, SA_EMAIL)

    h = Heatmap(
        layout_path=args.layout,
        threaded=args.threaded,
        debug=args.debug,
    )
    h.update_heatmap()
    h.plot_heatmap(blocking=True, show=True)


if __name__ == '__main__':
    main()
