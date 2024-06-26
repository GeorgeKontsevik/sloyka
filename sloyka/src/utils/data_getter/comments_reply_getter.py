from datetime import datetime

import pandas as pd
import requests


class CommentsReply:
    def get_comments(self, owner_id, post_id, access_token):
        """
        Retrieves comments for a specific post from VK API.

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
            "v": "5.131",
            "extended": 1,
            "count": 100,
            "need_likes": 1,
        }

        comments = []

        response = requests.get(
            "https://api.vk.com/method/wall.getComments", params=params, timeout=60
        )
        data = response.json()

        if "response" in data:
            for item in data["response"]["items"]:
                item["date"] = datetime.utcfromtimestamp(item["date"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if "likes" in item:
                    item["likes_count"] = item["likes"]["count"]
                comments.append(item)
                if item["thread"]["count"] > 0:
                    params["comment_id"] = item["id"]
                    subcomments = self.get_subcomments(
                        owner_id, post_id, access_token, params
                    )
                    comments.extend(subcomments)

        return comments

    def get_subcomments(self, owner_id, post_id, access_token, params):
        """
        Retrieves subcomments for a given post from VK API.

        Parameters:
            owner_id (int): The ID of the owner of the post.
            post_id (int): The ID of the post.
            access_token (str): The access token for making the API request.
            params (dict): Additional parameters for the API request.

        Returns:
            list: A list of subcomments for the given post.
        """
        subcomments = []

        response = requests.get(
            "https://api.vk.com/method/wall.getComments", params=params
        )
        data = response.json()

        if "response" in data:
            for item in data["response"]["items"]:
                item["date"] = datetime.utcfromtimestamp(item["date"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if "likes" in item:
                    item["likes_count"] = item["likes"]["count"]
                subcomments.append(item)

        return subcomments

    def comments_to_dataframe(self, comments):
        """
        Generate a DataFrame from a list of comments.

        :param comments: list of comments to be converted into a DataFrame
        :return: DataFrame containing selected columns from the comments
        """
        df = pd.DataFrame(comments)
        df = df[["id", "date", "text", "post_id", "parents_stack", "likes_count"]]
        return df
