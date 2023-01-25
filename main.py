import os

import uvicorn


def main() -> None:
    """Main function running the app"""
    uvicorn.run('rest_image_tracker.api:app', host='0.0.0.0', port=int(os.getenv('PORT', '80')))


if __name__ == '__main__':
    main()
