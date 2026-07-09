import argparse
import os
import uuid
from datetime import datetime, timedelta, timezone
from ftplib import FTP, error_perm, error_proto, error_reply, error_temp
from typing import List, Literal

from larker import Larker

utc_plus_8 = timezone(timedelta(hours=8))

# internal user is admin
OE_LLM_INTERNAL_FTP_HOST = os.environ.get("OE_LLM_INTERNAL_FTP_HOST", "")
OE_LLM_INTERNAL_FTP_USR = os.environ.get("OE_LLM_INTERNAL_FTP_USR", "")
OE_LLM_INTERNAL_FTP_PSW = os.environ.get("OE_LLM_INTERNAL_FTP_PSW", "")
# outside user is admin
OE_LLM_OUTSIDE_FTP_HOST = os.environ.get("OE_LLM_OUTSIDE_FTP_HOST", "")
OE_LLM_OUTSIDE_FTP_USR = os.environ.get("OE_LLM_OUTSIDE_FTP_USR", "")
OE_LLM_OUTSIDE_FTP_PSW = os.environ.get("OE_LLM_OUTSIDE_FTP_PSW", "")


class FTPSync:
    def __init__(self, upload_paths: List[str]) -> None:
        self.internal_ftp = FTP(
            host=OE_LLM_INTERNAL_FTP_HOST,
            user=OE_LLM_INTERNAL_FTP_USR,
            passwd=OE_LLM_INTERNAL_FTP_PSW)
        self.outside_ftp = FTP(
            host=OE_LLM_OUTSIDE_FTP_HOST,
            user=OE_LLM_OUTSIDE_FTP_USR,
            passwd=OE_LLM_OUTSIDE_FTP_PSW)
        self.upload_paths = upload_paths

    def check_exists(self,
                     ftp_type: Literal['internal', 'outside'],
                     check_path: str):
        if not check_path.startswith('openexplorer_llm/'):
            check_path = 'openexplorer_llm/' + check_path
        if ftp_type == "internal":
            ftp = self.internal_ftp
        else:
            ftp = self.outside_ftp
        ftp.cwd('/')
        parts = check_path.split('/')
        directory = '/'.join(parts[:-1])
        filename = parts[-1]
        try:
            ftp.cwd('/' + directory if directory else '/')
            files = ftp.nlst()
            if filename in files:
                return True
        except error_perm:
            return False

    def check(self) -> None:
        for upload_path in self.upload_paths:
            assert self.check_exists(
                "internal", upload_path), f"{upload_path} is not exists in inernal server"  # noqa
            assert not self.check_exists(
                "outside", upload_path), f"{upload_path} is exists in outside server"  # noqa

    def is_file(self, upload_path: str) -> bool:
        self.internal_ftp.cwd('/')
        try:
            self.internal_ftp.size(upload_path)
            return True
        except (error_perm, error_reply, error_temp, error_proto):
            try:
                self.internal_ftp.cwd(upload_path)
                self.internal_ftp.cwd('..')
                return False
            except (error_perm, error_reply, error_temp, error_proto):
                return False

    def sync_dir(self, upload_path: str) -> None:
        remote_dir, tail_name = os.path.split(upload_path)
        tmp_dir = f"tmp_{str(uuid.uuid4())}"
        os.mkdir(tmp_dir)
        # download
        download_cmd = f"""
            cd {tmp_dir} && lftp -u {OE_LLM_INTERNAL_FTP_USR},{OE_LLM_INTERNAL_FTP_PSW} {OE_LLM_INTERNAL_FTP_HOST} -e "mirror --verbose {upload_path} ./{tail_name};exit"
        """  # noqa
        print("download command", download_cmd)
        assert os.system(download_cmd) == 0, f"download {upload_path} failed"
        local_path = os.path.join(tmp_dir, tail_name)
        assert os.path.exists(local_path), f"{local_path} is not exists"
        # create remote dir
        if not self.check_exists("outside", remote_dir):
            create_dir_command = f"""
        cd {tmp_dir} && lftp -u {OE_LLM_OUTSIDE_FTP_USR},{OE_LLM_OUTSIDE_FTP_PSW} {OE_LLM_OUTSIDE_FTP_HOST} -e "mkdir -p {remote_dir};exit"
            """  # noqa
            print("create dir command", create_dir_command)
            assert os.system(
                create_dir_command) == 0, f"create {remote_dir} failed"
        # upload
        upload_command = f"""
    cd {tmp_dir} && lftp -u {OE_LLM_OUTSIDE_FTP_USR},{OE_LLM_OUTSIDE_FTP_PSW} {OE_LLM_OUTSIDE_FTP_HOST} -e "mirror -R {tail_name} {upload_path};exit"
        """  # noqa
        print("upload command", upload_command)
        assert os.system(upload_command) == 0, f"upload {upload_path} failed"

    def sync_file(self, upload_path: str) -> None:
        remote_dir, tail_name = os.path.split(upload_path)
        tmp_dir = f"tmp_{str(uuid.uuid4())}"
        os.mkdir(tmp_dir)
        # download
        download_cmd = f"""
        cd {tmp_dir} && lftp -u {OE_LLM_INTERNAL_FTP_USR},{OE_LLM_INTERNAL_FTP_PSW} {OE_LLM_INTERNAL_FTP_HOST} -e "get {upload_path};exit"
        """  # noqa
        print("download command", download_cmd)
        assert os.system(download_cmd) == 0, f"download {upload_path} failed"
        local_path = os.path.join(tmp_dir, tail_name)
        assert os.path.exists(local_path), f"{local_path} is not exists"
        # create remote dir
        if not self.check_exists("outside", remote_dir):
            create_dir_command = f"""
            cd {tmp_dir} && lftp -u {OE_LLM_OUTSIDE_FTP_USR},{OE_LLM_OUTSIDE_FTP_PSW} {OE_LLM_OUTSIDE_FTP_HOST} -e "mkdir -p {remote_dir};exit"
            """  # noqa
            print("create dir command", create_dir_command)
            assert os.system(create_dir_command) == 0, f"create {remote_dir} failed"  # noqa
        # upload
        upload_command = f"""
        cd {tmp_dir} && lftp -u {OE_LLM_OUTSIDE_FTP_USR},{OE_LLM_OUTSIDE_FTP_PSW} {OE_LLM_OUTSIDE_FTP_HOST} -e "cd {remote_dir};put {tail_name};exit"
        """  # noqa
        print("upload command", upload_command)
        assert os.system(upload_command) == 0, f"upload {upload_path} failed"
        assert self.check_exists("outside", upload_path), f"upload {upload_path} failed"  # noqa

    def sync(self) -> None:
        for upload_path in self.upload_paths:
            if not upload_path.startswith('openexplorer_llm/'):
                upload_path = 'openexplorer_llm/' + upload_path
            if self.is_file(upload_path):
                self.sync_file(upload_path)
            else:
                self.sync_dir(upload_path)

    def send_msg(self) -> None:
        larker = Larker()
        user_id = larker.get_user_id_by_name(
            os.environ.get("TRIGGER_USER", "sicong01.li"))
        chat_id = larker.get_chat_id_by_name("OE-LLM MR Review")
        date_now = datetime.now(tz=utc_plus_8).strftime("%Y-%m-%d %H:%M")
        sync_list = [{"sync_path": path} for path in self.upload_paths]
        larker.send_message_card_batch(
            template_id="AAq7UNcL4p9BD",
            template_variable={
                "sync_list": sync_list,
                "users": [user_id],
                "start_time": date_now},
            receive_dict={"user_id": [user_id], "chat_id": [chat_id]})

    def run(self) -> None:
        self.check()
        self.sync()
        self.send_msg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--upload_paths", help="upload paths")
    sync = FTPSync(upload_paths=parser.parse_args(
    ).upload_paths.strip().split(","))
    sync.run()
