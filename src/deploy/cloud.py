from mega import Mega
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
"""
https://www.geeksforgeeks.org/how-to-use-mega-nz-api-with-python/
https://github.com/odwyersoftware/mega.py/blob/master/src/mega/mega.py
"""
def create_subdir(mega, dirname, dest):
    mega.create_folder(
        dirname,
        dest=dest
    )
    newdir_path = mega.find(dirname)[0]
    return newdir_path

def upload_file(mega, src, dest):
    status = mega.upload(
        src,
        dest
    )
    return mega.get_upload_link(status)

def main(mega, local_root, cloud_root):
    ## browse the local files
    print('Start uploading...')
    needed_uploads = os.listdir(local_root)
    for item in tqdm(needed_uploads):
        item_local = os.path.join(local_root, item)
        if os.path.isdir(item_local):
            ## create the subdir on cloud
            dir_exist = mega.find(item)
            if not dir_exist:
                print(f"Creating dir: {item}")
                item_cloud = create_subdir(
                    mega=mega,
                    dirname=item,
                    dest=cloud_root
                )
            else:
                item_cloud = dir_exist[0]

            files_in_local = os.listdir(item_local)
            files_local_abs_path = [
                os.path.join(item_local, item) for item in files_in_local
            ]
            for f in files_local_abs_path:
                fname = f.split('\\')[-1]
                file_exist = mega.find(fname)
                ## check existing
                if not file_exist:
                    try:
                        file_in_cloud = upload_file(
                            mega=mega,
                            src=f,
                            dest=item_cloud
                        )
                    except Exception as e:
                        print(f'Error: {e}')
                else:
                    # print(f'File {fname} exists')
                    continue

if __name__ == "__main__":

    local_root = r"D:\data\fashion"
    mega = Mega()
    mega = mega.login(os.getenv("mega_email"), os.getenv("mega_password"))
    cloud_root = mega.find("RecSys")[0]
    main(mega, local_root, cloud_root)


    