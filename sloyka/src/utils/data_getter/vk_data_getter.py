"""
This module contains classes for retrieving and working with various types of data.

@class:GeoDataGetter:
This class is used to retrieve geospatial data from OpenStreetMap (OSM) based on given OSM ID and tags.

@class:VKParser:
A class for parsing and working with VK comments and posts. Combines posts and comments into one dataframe.

@class:Streets:
A class for working with street data.

"""

import datetime
import random
import time

import pandas as pd
import requests
from tqdm import tqdm


class VKParser:
    API_VERISON = "5.131"
    COUNT_ITEMS = 100
    # SLEEP_TIME = 0.5
    TIMEOUT_LIMIT = 15

    @staticmethod
    def get_group_name(domain, accsess_token):
        params = {
            "group_id": domain,
            "access_token": accsess_token,
            "v": VKParser.API_VERISON,
        }
        response = requests.get(
            "https://api.vk.com/method/groups.getById", params=params
        )  # передвинуть повыше
        data = response.json()
        if "response" in data and data["response"]:
            group_name = data["response"][0]["name"]
            return pd.DataFrame({"group_name": [group_name]})
        else:
            print("Error while fetching group name:", data)
            return pd.DataFrame({"group_name": [None]})

    @staticmethod
    def get_owner_id_by_domain(domain, access_token):
        """
        Get the owner ID of a VK group by its domain.

        Args:
            domain (str): The domain of the VK group.
            access_token (str): The access token for the VK API.

        Returns:
            int: The owner ID of the VK group, or None if the request was not successful.
        """
        url = "https://api.vk.com/method/wall.get"
        params = {
            "domain": domain,
            "access_token": access_token,
            "v": VKParser.API_VERISON,
        }
        response = requests.get(url, params=params)
        if response.ok:
            owner_id = response.json()["response"]["items"][0]["owner_id"]
        else:
            owner_id = None
        return owner_id

    @staticmethod
    def get_group_post_ids(domain, access_token, post_num_limit, step) -> list:
        """
        A static method to retrieve a list of post IDs for a given group, based on the owner ID,
        access token, post number limit, and step size. Returns a list of post IDs.
        """
        offset = 0
        post_ids = []

        while offset < post_num_limit:
            print(offset, " | ", post_num_limit, end="\r")
            res = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": access_token,
                    "v": VKParser.API_VERISON,
                    "domain": domain,
                    "count": step,
                    "offset": offset,
                },
                timeout=10,
            ).json()["response"]
            # print(res.json().keys())
            time.sleep(random.random())

            post_ids_new = [k["id"] for k in res["items"]]
            post_ids += post_ids_new
            offset += step

        return post_ids

    @staticmethod
    def get_subcomments(owner_id, post_id, access_token, params):
        """
        Retrieves subcomments from the VK API.

        Args:
            owner_id (int): The ID of the owner of the comments.
            post_id (int): The ID of the post.
            access_token (str): The access token for authentication.
            params (dict): Additional parameters for the API request.

        Returns:
            list: A list of subcomments retrieved from the API.
        """
        subcomments = []

        response = requests.get(
            "https://api.vk.com/method/wall.getComments", params=params
        )
        # print(response.json().keys())
        time.sleep(random.random())
        data = response.json()

        if "response" in data:
            for item in data["response"]["items"]:
                item["date"] = datetime.datetime.utcfromtimestamp(
                    item["date"]
                ).strftime("%Y-%m-%d %H:%M:%S")
                if "likes" in item:
                    item["likes.count"] = item["likes"]["count"]
                subcomments.append(item)

        return subcomments

    def get_comments(owner_id, post_id, access_token):
        """
        Get comments for a post on VK using the specified owner ID, post ID, and access token.

        Parameters:
            owner_id (int): The ID of the post owner.
            post_id (int): The ID of the post.
            access_token (str): The access token for authentication.

        Returns:
            list: A list of dictionaries containing comment information.
        """
        params = {
            "owner_id": owner_id,
            "post_id": post_id,
            "access_token": access_token,
            "v": VKParser.API_VERISON,
            "extended": 1,
            "count": 100,
            "need_likes": 1,
        }

        comments = []

        response = requests.get(
            "https://api.vk.com/method/wall.getComments", params=params
        )
        # print(response.json().keys())
        time.sleep(random.random())
        data = response.json()

        if "response" in data:
            for item in data["response"]["items"]:
                if item["text"] == "":
                    continue
                item["date"] = datetime.datetime.utcfromtimestamp(
                    item["date"]
                ).strftime("%Y-%m-%d %H:%M:%S")
                if "likes" in item:
                    item["likes.count"] = item["likes"]["count"]
                comments.append(item)
                if item["thread"]["count"] > 0:
                    params["comment_id"] = item["id"]
                    subcomments = VKParser.get_subcomments(
                        owner_id, post_id, access_token, params
                    )
                    comments.extend(subcomments)
        return comments

    @staticmethod
    def comments_to_dataframe(comments):
        """
        Convert comments to a DataFrame.

        Args:
            comments: List of comments to be converted.

        Returns:
            DataFrame: A DataFrame containing specific columns from the input comments.
        """
        df = pd.DataFrame(comments)
        df = df[
            ["id", "from_id", "date", "text", "post_id", "parents_stack", "likes.count"]
        ]
        return df

    @staticmethod
    def run_posts(
        domain, access_token, cutoff_date, number_of_messages=float("inf"), step=50
    ):
        """
        A function to retrieve posts from a social media API based on specified parameters.

        Parameters:
            owner_id (int): The ID of the owner whose posts are being retrieved.
            access_token (str): The authentication token for accessing the API.
            step (int): The number of posts to retrieve in each API call.
            cutoff_date (str): The date to stop retrieving posts (format: '%Y-%m-%d').
            number_of_messages (float): The maximum number of messages to retrieve (default is infinity).

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved posts.
        """

        domain = domain
        offset = 0
        all_posts = []
        if step > number_of_messages:
            step = number_of_messages
        while offset < number_of_messages:
            print(offset, " | ", number_of_messages, end="\r")

            response = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": access_token,
                    "v": VKParser.API_VERISON,
                    "domain": domain,
                    "count": step,
                    "offset": offset,
                },
                timeout=600,
            )
            if response.ok:
                # print(response.json().keys())
                data = response.json()["response"]["items"]
                offset += step
                current_posts = pd.json_normalize(data)
                current_posts = current_posts[
                    [
                        "date",
                        "id",
                        "text",
                        "views.count",
                        "likes.count",
                        "reposts.count",
                    ]
                ]
                current_posts["date"] = [
                    datetime.datetime.fromtimestamp(current_posts["date"][i])
                    for i in range(len(current_posts["date"]))
                ]
                current_posts["type"] = "post"
                all_posts.append(current_posts)
                print(current_posts.date.min())
                if any(
                    current_posts["date"]
                    < datetime.datetime.strptime(cutoff_date, "%Y-%m-%d")
                ):
                    print("posts downloaded")
                    break
            else:
                continue
            time.sleep(random.random())
        df_posts = pd.concat(all_posts).reset_index(drop=True)
        df_posts = df_posts[df_posts.text.map(lambda x: len(x)) > 0]
        df_posts["text"] = df_posts["text"].str.replace(r"\n", "", regex=True)
        df_posts["link"] = df_posts["text"].str.extract(r"(https://\S+)")
        return df_posts

    @staticmethod
    def run_comments(domain, post_ids, access_token, n_attempts=5):
        """
        Retrieves comments for multiple posts specified by their IDs and returns them as a DataFrame.

        Parameters:
            domain (str): The domain or group name.
            post_ids (list): List of post IDs to retrieve comments for.
            access_token (str): The access token for authentication.
            n_attempts (int): The number of retry attempts in case of failures. Defaults to 5.

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved comments with additional metadata.
        """
        owner_id = VKParser.get_owner_id_by_domain(domain, access_token)
        all_comments = []
        for post_id in tqdm(post_ids):
            attempts = 0
            while attempts < n_attempts:
                try:
                    comments = VKParser.get_comments(owner_id, post_id, access_token)
                    all_comments.extend(comments)
                    break  # Exit the retry loop if successful
                except requests.exceptions.Timeout:
                    print(f"Timeout error occurred for post_id {post_id}. Retrying...")
                    attempts += 1
                    time.sleep(2**attempts)  # Exponential backoff
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}. Retrying...")
                    attempts += 1
                    time.sleep(2**attempts)
            else:
                print(f"Failed to retrieve comments for post_id {post_id}.")

        df = VKParser.comments_to_dataframe(all_comments)
        df["type"] = "comment"
        df = df.reset_index(drop=True)
        print("comments downloaded")
        return df

    @staticmethod
    def run_parser(
        domain, access_token, cutoff_date, number_of_messages=float("inf"), step=100
    ):
        """
        Runs the parser with the given parameters and returns a combined DataFrame of posts and comments.

        :param owner_id: The owner ID for the parser.
        :param access_token: The user token for authentication.
        :param step: The step size for fetching data.
        :param cutoff_date: The cutoff date for fetching data.
        :param number_of_messages: The maximum number of messages to fetch. Defaults to positive infinity.
        :return: A combined DataFrame of posts and comments.
        """
        owner_id = VKParser.get_owner_id_by_domain(domain, access_token)
        df_posts = VKParser.run_posts(
            domain=owner_id,
            access_token=access_token,
            step=step,
            cutoff_date=cutoff_date,
            number_of_messages=number_of_messages,
        )
        post_ids = df_posts["id"].tolist()

        df_comments = VKParser.run_comments(
            domain=owner_id, post_ids=post_ids, access_token=access_token
        )
        df_comments.loc[
            df_comments["parents_stack"].apply(lambda x: len(x) > 0), "type"
        ] = "reply"
        for i in range(len(df_comments)):
            tmp = df_comments["parents_stack"].iloc[i]
            if tmp is not None:
                if len(tmp) > 0:
                    df_comments["parents_stack"].iloc[i] = tmp[0]
                else:
                    df_comments["parents_stack"].iloc[i] = None

        df_combined = df_comments.join(df_posts, on="post_id", rsuffix="_post")
        df_combined = pd.concat([df_posts, df_comments], ignore_index=True)
        df_group_name = VKParser.get_group_name(domain, access_token)
        df_combined["group_name"] = df_group_name["group_name"][0]

        return df_combined
