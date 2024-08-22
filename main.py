import os

import time
import requests
import pandas as pd

ORG = "xarray-contrib"
REPO = "xskillscore"
LLM_CHAT_FRAMEWORK = "openai"
EMBEDDINGS_FRAMEWORK = "openai"
LLM_AGENT_FRAMEWORK = "google"
GEOLOCATER_FRAMEWORK = "photon"
SNAPSHOT_FOLDER = "snapshot_2024-08-22"
CREATED_AFTER_DATE = pd.Timestamp("2000-01-01", tz="UTC")

BOTS = [
    "GPUtester",
    "codecov[bot]",
    "codecov-commenter",
    "codecov-io",
    "dependabot[bot]",
    "github-actions[bot]",
    "pep8speaks",
    "pre-commit-ci[bot]",
    "review-notebook-app[bot]",
]

ISSUE_PR_COLUMNS = [
    "author_association",
    "body",
    "comments",
    "created_at",
    "label_names",
    "number",
    "reactions.+1",
    "reactions.-1",
    "reactions.confused",
    "reactions.eyes",
    "reactions.heart",
    "reactions.hooray",
    "reactions.laugh",
    "reactions.rocket",
    "reactions.total_count",
    "state",
    "title",
    "url",
    "user.login",
]

USER_COLUMNS = [
    "avatar_url",
    "bio",
    "blog",
    "company",
    "created_at",
    "email",
    "followers",
    "following",
    "html_url",
    "location",
    "location_lat",
    "location_lon",
    "login",
    "name",
    "twitter_username",
    "updated_at",
]

COMMENT_COLUMNS = [
    "body",
    "created_at",
    "user.login",
]

ISSUE_COLUMNS = [
    "number",
    "issue_body",
    "issue_n_comments",
    "issue_created_at",
    "issue_label_names",
    "issue_reactions.total_count",
    "issue_title",
    "issue_user_login",
    "issue_user_company",
]

PR_COLUMNS = [
    "number",
    "pr_body",
    "pr_n_comments",
    "pr_created_at",
    "pr_label_names",
    "pr_reactions.total_count",
    "pr_title",
    "pr_user_login",
    "pr_user_company",
]

try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
except KeyError:
    print("env var OPENAI_API_KEY not found")
    OPENAI_API_KEY = ""
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
except KeyError:
    print("env var GOOGLE_API_KEY not found")
    GOOGLE_API_KEY = ""
try:
    GITHUB_API_TOKEN = os.environ.get("GITHUB_API_TOKEN")
    HEADERS = {"Authorization": f"token {GITHUB_API_TOKEN}"}
except KeyError:
    print("env var GITHUB_API_TOKEN not found")
    GITHUB_API_TOKEN = ""
    HEADERS = {}


def _get_with_retry(url):
    for attempt in range(3):
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response
        elif response.status_code in [403, 429]:
            print(
                f"Rate limit hit (status code: {response.status_code}). Retrying in 1 hour"
            )
            time.sleep(3600)
        else:
            print(f"Unexpected status code: {response.status_code}")
            return response
    print(f"Max retries reached. Last status code: {response.status_code}")
    return response


def _geocode_with_retry(location, max_retries=3, timeout=30):
    from geopy.exc import GeocoderUnavailable
    from time import sleep
    from geopy.geocoders import Photon

    geolocator = Photon()

    for attempt in range(max_retries):
        try:
            geocoded_location = geolocator.geocode(location, timeout=timeout)
            if geocoded_location:
                return geocoded_location
            else:
                print(f"Location not found: {location}")
                return None
        except GeocoderUnavailable:
            if attempt < max_retries - 1:
                sleep_time = 2**attempt
                print(
                    f"Geocoding attempt {attempt + 1} failed. Retrying in {sleep_time} seconds..."
                )
                sleep(sleep_time)
            else:
                print(
                    f"Geocoding failed after {max_retries} attempts for location: {location}"
                )
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return None


def _json_content_check(json_content) -> bool:
    if not json_content:
        print("no content found in response")
        return False
    else:
        return True


def _chat_response(
    llm_chat_framework=LLM_CHAT_FRAMEWORK,
    content="Why is the sky blue?",
    api_key=OPENAI_API_KEY,
):
    if llm_chat_framework == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content
    elif llm_chat_framework == "google":
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = client.generate_content(content)
        return response.text
    else:
        raise ValueError("Unsupported LLM chat framework")


def _num_tokens_from_string(
    string: str,
    encoding_name: str = "cl100k_base",
) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _agent_response(agent, content):
    return agent.invoke(content)["output"]


def scrape_gh(
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
    verbose: bool = False,
) -> None:
    """
    Puts data into eight folders:
    open_issues, closed_issues, open_prs, closed_prs
    These contain the titles and body

    open_issues_comments, closed_issues_comments, open_prs_comments, closed_prs_comments
    These contain the comments.

    GitHub shares the same structure for issues and PRs
    Note: not tested for only open issues for example
    """
    if GITHUB_API_TOKEN == "":
        raise ValueError("GITHUB_API_TOKEN not set")
    if verbose:
        print(f"{states=}, {content_types=}")
    import json
    import pandas as pd

    from tqdm.auto import tqdm

    GH_API_URL_PREFIX = f"https://api.github.com/repos/{ORG}/{REPO}/"

    users = set()
    for state in states:
        if verbose:
            print(f"{state=}")
        page = 1
        while True:
            # the issues endpoint is misnomer and contains issues and prs.
            # This returns a high level overview of the issue or pr such as:
            # the user, the body, body reactions e.g.
            # +1 and whether it's a pr or issue
            gh_api_url_suffix = f"issues?state={state}&per_page=100&page={page}"
            if verbose:
                print(f"{gh_api_url_suffix=}")
            url = f"{GH_API_URL_PREFIX}{gh_api_url_suffix}"
            response = _get_with_retry(url)
            # list of ~100 issues or prs from most recent to oldest
            page_issues_or_prs = response.json()
            if not _json_content_check(page_issues_or_prs):
                break
            # Exlude bots
            page_issues_or_prs = [
                page_issue_or_pr
                for page_issue_or_pr in page_issues_or_prs
                if page_issue_or_pr["user"]["login"] not in BOTS
            ]
            # keep issues or prs after CREATED_AFTER_DATE
            page_issues_or_prs = [
                page_issue_or_pr
                for page_issue_or_pr in page_issues_or_prs
                if pd.Timestamp(page_issue_or_pr["created_at"]) >= CREATED_AFTER_DATE
            ]
            if not page_issues_or_prs:
                break
            for content_type in content_types:
                if verbose:
                    print(f"{content_type=}")
                folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}"
                os.makedirs(folder, exist_ok=True)
                comment_folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}_comments"
                os.makedirs(comment_folder, exist_ok=True)
                if content_type == "issues":
                    endpoint = "issues"
                    page_issues_or_prs_filtered = [
                        issue
                        for issue in page_issues_or_prs
                        if "pull_request" not in issue
                    ]
                elif content_type == "prs":
                    endpoint = "pulls"
                    page_issues_or_prs_filtered = [
                        pr for pr in page_issues_or_prs if "pull_request" in pr
                    ]
                else:
                    raise ValueError(
                        f"Unknown content type: {content_type}. "
                        "Should be 'issues' or 'prs'"
                    )

                for issue_or_pr in tqdm(
                    page_issues_or_prs_filtered,
                    f"fetching {state} {content_type} for page {page}",
                ):
                    number = issue_or_pr["number"]
                    padded_number = f"{number:06d}"
                    filename = (
                        f"{folder}/{content_type[:-1]}_detail_{padded_number}.json"
                    )
                    detail_url = f"{GH_API_URL_PREFIX}{endpoint}/{number}"
                    if verbose:
                        print(f"{detail_url=}")
                    detail_response = _get_with_retry(detail_url)
                    detail_response_json = detail_response.json()
                    if not _json_content_check(detail_response_json):
                        break
                    # There is also a timeline API that could be included.
                    # This contains information on cross posting issues or prs

                    # Reactions for PRs can be found in the issue endpoint
                    if content_type == "prs":
                        detail_url2 = f"{GH_API_URL_PREFIX}issues/{number}"
                        if verbose:
                            print(f"{detail_url2=}")
                        detail_response2 = _get_with_retry(detail_url2)
                        detail_response2_json = detail_response2.json()
                        if not _json_content_check(detail_response2_json):
                            break
                        detail_response_json["reactions"] = detail_response2_json[
                            "reactions"
                        ]
                    with open(filename, "w") as f:
                        json.dump(detail_response_json, f, indent=4)
                    # Save the user who create the issue or PR
                    users.add(detail_response_json["user"]["login"])
                    # Get comments data if comments (which is n_comments) > 0
                    if detail_response_json["comments"] > 0:
                        filename = f"{comment_folder}/" f"comments_{padded_number}.json"
                        comments_url = f"{GH_API_URL_PREFIX}issues/{number}/comments"
                        if verbose:
                            print(f"{comments_url=}")
                        comments_response = _get_with_retry(comments_url)
                        comments_response_json = comments_response.json()
                        if not _json_content_check(comments_response_json):
                            break
                        # Exclude bots in comments
                        comments_response_json = [
                            comment
                            for comment in comments_response_json
                            if comment["user"]["login"] not in BOTS
                        ]
                        if len(comments_response_json) > 1:
                            with open(filename, "w") as f:
                                json.dump(comments_response_json, f, indent=4)
                            # Save the users who have commented on the issue or pr
                            for comment in comments_response_json:
                                users.add(comment["user"]["login"])
            page += 1

    # Scrape users
    users_list = list(users)
    folder = f"{SNAPSHOT_FOLDER}/users"
    os.makedirs(folder, exist_ok=True)
    for username in tqdm(users_list, "fetching data for users"):
        user_url = f"https://api.github.com/users/{username}"
        if verbose:
            print(f"{user_url=}")
        user_detail_response = _get_with_retry(user_url)
        user_detail = user_detail_response.json()
        if not _json_content_check(detail_response2_json):
            break
        file_path = os.path.join(folder, f"user_detail_{username}.json")
        with open(file_path, "w") as f:
            json.dump(user_detail, f, indent=4)
    return None


def create_df(
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
) -> None:
    """
    Concat data into a dataframe.

    Row UUID is content_type number + comment
    """
    import json
    import os
    import pandas as pd

    from tqdm.auto import tqdm

    # Create a user dataframe
    folder = f"{SNAPSHOT_FOLDER}/users"
    files = sorted(os.listdir(folder))
    users_df = pd.DataFrame()
    for file in tqdm(files, "concatenating users"):
        with open(f"{folder}/{file}", "r") as f:
            data = json.load(f)
        _df = pd.json_normalize(data)
        # geolocate users
        if GEOLOCATER_FRAMEWORK != "None":
            if GEOLOCATER_FRAMEWORK == "photon":
                geocoded_location = _geocode_with_retry(_df["location"][0])
                if geocoded_location is not None:
                    _df["location_lat"] = geocoded_location.latitude
                    _df["location_lon"] = geocoded_location.longitude
                else:
                    _df["location_lat"] = None
                    _df["location_lon"] = None
            else:
                raise ValueError(
                    f"Unsupported geolocator framework: {GEOLOCATER_FRAMEWORK}"
                )
        for col in ["created_at", "updated_at"]:
            _df[col] = pd.Timestamp(_df[col][0])
        _df = _df[USER_COLUMNS]
        users_df = pd.concat([users_df, _df], ignore_index=True)
    users_df = users_df.add_prefix("user_")
    users_df.to_csv(f"{SNAPSHOT_FOLDER}/users.csv", index=False)
    users_df.to_parquet(f"{SNAPSHOT_FOLDER}/users.parquet")

    for state in states:
        for content_type in content_types:
            folder = f"{SNAPSHOT_FOLDER}/{state}_{content_type}"
            files = sorted(os.listdir(folder))
            df = pd.DataFrame()
            for file in tqdm(files, f"concatenating {state} {content_type}"):
                padded_number = file.split("_")[-1].split(".")[0]
                with open(f"{folder}/{file}", "r") as f:
                    data = json.load(f)
                _df = pd.json_normalize(data)
                # You can open or a issue or PR just just a title
                if _df["body"][0] is None:
                    _df["body"] = ""
                _df["label_names"] = _df["labels"].apply(
                    lambda x: [label["name"] for label in x]
                    if isinstance(x, list)
                    else []
                )
                _df["created_at"] = pd.Timestamp(_df["created_at"][0])
                _df["url"] = _df["url"][0].replace(
                    "https://api.github.com/repos/", "https://github.com/"
                )
                if LLM_CHAT_FRAMEWORK != "None":
                    _df["LLM_title_subject"] = _chat_response(
                        LLM_CHAT_FRAMEWORK,
                        "Give me a one word summary of the following GitHub "
                        f"{REPO} {content_type[:-1]} title: {_df['title'][0]}",
                    )
                    _df = _df[ISSUE_PR_COLUMNS + ["LLM_title_subject"]]
                else:
                    _df = _df[ISSUE_PR_COLUMNS]
                _df = _df.rename(
                    columns={"comments": "n_comments", "user.login": "user_login"}
                )
                _df = _df.add_prefix(f"{content_type[:-1]}_")
                _df["number"] = int(padded_number)

                # Read comment data if exists
                comment_file = (
                    f"{SNAPSHOT_FOLDER}/{state}_{content_type}_comments/"
                    f"comments_{padded_number}.json"
                )
                if os.path.exists(comment_file):
                    with open(comment_file, "r") as f:
                        data = json.load(f)
                    _df2 = pd.json_normalize(data)
                    _df2["created_at"] = pd.to_datetime(_df2["created_at"])
                    _df2 = _df2[COMMENT_COLUMNS].copy()
                    _df2 = _df2.rename(
                        columns={
                            "user.login": "user_login",
                        }
                    )
                    _df2 = _df2.add_prefix(f"{content_type[:-1]}_comment_")
                    _df2["number"] = int(padded_number)
                else:
                    # empty comment df
                    _df2 = pd.DataFrame(
                        {
                            f"{content_type[:-1]}_comment_body": [""],
                            f"{content_type[:-1]}_comment_created_at": [
                                pd.Timestamp(
                                    "2000-01-01", tz="UTC"
                                )
                            ],
                            f"{content_type[:-1]}_comment_user_login": [""],
                        }
                    )
                    _df2["number"] = int(padded_number)
                # Join comments with issue/pr
                _df = pd.merge(_df, _df2, on="number", how="left")
                # Geta info about poster
                _df = pd.merge(
                    _df,
                    users_df,
                    left_on=f"{content_type[:-1]}_user_login",
                    right_on="user_login",
                    how="left",
                )
                for col in USER_COLUMNS:
                    _df[f"{content_type[:-1]}_user_{col}"] = _df[f"user_{col}"]
                    del _df[f"user_{col}"]
                # Get info about commenters
                _df = _df.merge(
                    users_df,
                    left_on=f"{content_type[:-1]}_comment_user_login",
                    right_on="user_login",
                    how="left",
                )
                for col in USER_COLUMNS:
                    _df[f"{content_type[:-1]}_comment_user_{col}"] = _df[f"user_{col}"]
                    del _df[f"user_{col}"]

                df = pd.concat([df, _df], axis=0).reset_index(drop=True)
            df.to_csv(f"{SNAPSHOT_FOLDER}/{state}_{content_type}.csv", index=False)
            df.to_parquet(f"{SNAPSHOT_FOLDER}/{state}_{content_type}.parquet")
    return None


def create_vector_db(
    states: list[str] = ["open", "closed"],
    content_types: list[str] = ["issues", "prs"],
) -> None:
    if EMBEDDINGS_FRAMEWORK == "None":
        raise ValueError("Embeddings framework cannot be None")
    import os
    import pickle
    import pandas as pd
    from langchain_openai import OpenAIEmbeddings
    from pymilvus import MilvusClient

    if os.path.exists(f"{SNAPSHOT_FOLDER}/milvus.db"):
        os.remove(f"{SNAPSHOT_FOLDER}/milvus.db")
    client = MilvusClient(f"{SNAPSHOT_FOLDER}/milvus.db")
    for state in states:
        for content_type in content_types:
            print(f"Creating vector db for {state} {content_type}")
            df = pd.read_parquet(f"{SNAPSHOT_FOLDER}/{state}_{content_type}.parquet")
            if not df.empty:
                if content_type == "issues":
                    _df = df[
                        ISSUE_COLUMNS + [f"{content_type[:-1]}_LLM_title_subject"]
                    ].copy()
                else:
                    _df = df[
                        PR_COLUMNS + [f"{content_type[:-1]}_LLM_title_subject"]
                    ].copy()
                _df[f"{content_type[:-1]}_label_names"] = _df[
                    f"{content_type[:-1]}_label_names"
                ].apply(tuple)
                _df = _df.drop_duplicates().reset_index(drop=True)
                if EMBEDDINGS_FRAMEWORK == "openai":
                    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                else:
                    raise ValueError(
                        f"Unsupported embeddings framework: {EMBEDDINGS_FRAMEWORK}"
                    )
                embeddings = embeddings_model.embed_documents(
                    _df[f"{content_type[:-1]}_body"].fillna("").values
                )  # ndocs x 1536
                with open(
                    f"{SNAPSHOT_FOLDER}/{state}_{content_type}_embeddings.pkl", "wb"
                ) as f:
                    pickle.dump(embeddings, f)
                data = [
                    {
                        "id": row["number"],
                        "vector": embeddings[i],
                        "text": row[f"{content_type[:-1]}_body"],
                        "subject": row[f"{content_type[:-1]}_LLM_title_subject"],
                    }
                    for i, row in _df.iterrows()
                ]

                client.create_collection(
                    collection_name=f"{state}_{content_type}",
                    dimension=1536,
                )
                _ = client.insert(
                    collection_name=f"{state}_{content_type}",
                    data=data,
                )
    return None


def st_dashboard():
    from streamlit_folium import st_folium
    import pandas as pd
    import folium
    from pymilvus import MilvusClient
    from folium import CustomIcon
    from langchain_openai import OpenAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import OpenAI as OpenAI_langchain
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title(f"{REPO} GitHub explorer")

    st.markdown(
        """
    This dashboard can help with a variety of personas:
    - Product Manager: Identification of users/leads.
    - Developer Advocate / Solutions Architect: Identify common developer pain points.
    - Software Engineer: Help develop the roadmap.
    - Project Manager: Identify most common requested features.
    - User: Indentify other companies who use this product. Are they hiring?
    """
    )

    st.subheader("Partners")

    df_users = pd.read_parquet(f"{SNAPSHOT_FOLDER}/users.parquet")

    content_type = st.selectbox("content:", ["issues", "prs"])
    status = st.selectbox("status:", ["open", "closed"])

    df = pd.read_parquet(f"{SNAPSHOT_FOLDER}/{status}_{content_type}.parquet")

    if df.empty:
        st.write("No data available for the selected criteria.")
    else:
        tab1, tab2 = st.tabs(["Posters", "Commenters"])
        with tab1:
            _df = df[
                [f"{content_type[:-1]}_user_login", f"{content_type[:-1]}_user_company"]
            ].drop_duplicates()
            _counts = _df[f"{content_type[:-1]}_user_company"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            _counts.plot(
                kind="bar",
                ax=ax,
                xlabel="Company",
                ylabel="Count",
                title=f"Count of company employees who create {content_type} in the {REPO} repo",
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
        with tab2:
            _df = df[
                [
                    f"{content_type[:-1]}_comment_user_login",
                    f"{content_type[:-1]}_comment_user_company",
                ]
            ].drop_duplicates()
            _counts = _df[f"{content_type[:-1]}_comment_user_company"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            _counts.plot(
                kind="bar",
                ax=ax,
                xlabel="Company",
                ylabel="Count",
                title=f"Count of companies employees who comment on {content_type} in the {REPO} repo",
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown(
        "Use the LLM below to ask questions such as:"
        " - **'What type of company is X?'** "
        "or generic questions such as:"
        f" - **'Why would company X use {REPO}?'** "
        "but don't expect great results. "
        "We will use the GitHub data to refine this question later. "
    )

    st.markdown("**You will need to pass an OpenAI API key to ask questions below:**")
    openai_api_key = st.text_input("OpenAI API Key:", type="password")
    content = st.text_input(
        f"Ask questions about companies who use {REPO} "
        "(replace X with a company listed above):",
        "What type of company is X?",
    )
    if openai_api_key:
        st.write(_chat_response("openai", content, openai_api_key))

    st.subheader("Community")

    st.markdown(
        "We can explore the location of users. "
        "This can help with event planning and community building. "
    )

    gdf = gpd.GeoDataFrame(
        df_users,
        geometry=gpd.points_from_xy(
            df_users["user_location_lon"],
            df_users["user_location_lat"],
        ),
        crs="epsg:4326",
    )
    m = gdf.explore()
    for idx, row in gdf.iterrows():
        icon = CustomIcon(
            icon_image=row["user_avatar_url"], icon_size=(30, 30), icon_anchor=(15, 15)
        )
        popup = f"""
        <b>{row['user_name']}</b><br>
        <b>Bio:</b> {row['user_bio']}<br>
        <b>Blog:</b> <a href="{row['user_blog']}" target="_blank">{row['user_blog']}</a><br>
        <b>Company:</b> {row['user_company']}<br>
        <b>Created at:</b> {row['user_created_at']}<br>
        <b>Email:</b> {row['user_email']}<br>
        <b>Followers:</b> {row['user_followers']}<br>
        <b>Following:</b> {row['user_following']}<br>
        <b>GitHub:</b> <a href="{row['user_html_url']}" target="_blank">{row['user_html_url']}</a><br>
        <b>Location:</b> {row['user_location']}<br>
        <b>Coordinates:</b> {row['user_location_lat']}, {row['user_location_lon']}<br>
        <b>Login:</b> {row['user_login']}<br>
        <b>Twitter:</b> {row['user_twitter_username'] if row['user_twitter_username'] else 'N/A'}<br>
        <b>Updated at:</b> {row['user_updated_at']}
        """
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup, max_width=300),
            icon=icon,
        ).add_to(m)
    st_folium(m, width=900)

    st.markdown("Click on a member to find out more information about them.")

    st.subheader("Users")

    st.markdown(
        """
        We can explore the GitHub data to understand what developers are interested in, 
        and to ensure their requested features or bug report are taken into account in the roadmap.
        You can ask questions such as: 
        - **What issues are company X most interested in?**
        - **What issue has the most total reactions?**
        - **What company posted the issue with the most reactions?**
        - **What are the top 5 issues with the most reactions?**
        """
    )

    st.markdown(
        "The limit rows is provided below to ensure you are able to fit the text "
        "into the API calls. You will get more accruate results with more data."
    )

    df = pd.read_parquet(f"{SNAPSHOT_FOLDER}/open_issues.parquet")
    _df = df[ISSUE_COLUMNS].copy()
    _df["issue_label_names"] = _df["issue_label_names"].apply(tuple)

    st_limit_rows = st.selectbox(
        "limit rows:", [10, 25, 50, 100, 250, 500, 1000, 10000]
    )

    _df_filtered = _df.drop_duplicates().head(st_limit_rows).reset_index(drop=True)

    st_llm_agent_framework = st.selectbox(
        "agent framework:", ["google", "openai", "None"]
    )

    created_agent = False
    if st_llm_agent_framework == "openai":
        if openai_api_key:
            agent = create_pandas_dataframe_agent(
                OpenAI_langchain(
                    temperature=0,
                    model="gpt-3.5-turbo-instruct",
                    openai_api_key=openai_api_key,
                ),
                _df_filtered,
                allow_dangerous_code=True,
                verbose=True,
            )
            created_agent = True
    elif st_llm_agent_framework == "google":
        google_api_key = st.text_input("Google API Key:", type="password")
        if google_api_key:
            agent = create_pandas_dataframe_agent(
                ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                ),
                _df_filtered,
                allow_dangerous_code=True,
                verbose=True,
            )
            created_agent = True

    content = st.text_input(
        "Ask questions about about users and developers such as:",
        "What issues has the company X created?",
    )
    if created_agent:
        response = _agent_response(agent, content)
        st.write(response)
        if ":" in response:
            response = response.split(":")[1].strip()

    if EMBEDDINGS_FRAMEWORK != "None":
        st.markdown(
            "We will now use a vector database to query matching issues. "
            "This can help first time posters find similar issues. "
            "(Requires OpenAI API key)"
        )
        if openai_api_key and created_agent:
            embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
            question = f"What issues are similar to {response}?"
            # Look up values for the vector db matching the question
            question_embeddings = embeddings_model.embed_documents([question])
            client = MilvusClient(f"{SNAPSHOT_FOLDER}/milvus.db")
            res = client.search(
                collection_name="open_issues",
                data=question_embeddings,
                limit=2,
            )
            similar_issue_1 = res[0][0]["id"]
            similar_issue_2 = res[0][1]["id"]
            _df1 = df[df["number"] == similar_issue_1]
            _df2 = df[df["number"] == similar_issue_2]
            st.dataframe(_df1)
            if not _df1.equals(_df2):
                st.dataframe(_df2)


def run_all(
    states=["open", "closed"],
    content_types=["issues", "prs"],
    verbose=False,
):
    scrape_gh(states=states, content_types=content_types, verbose=verbose)
    create_df(states=states, content_types=content_types)
    create_vector_db(states=states, content_types=content_types)
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        st_dashboard()
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("function", type=str, help="Function to call")
        parser.add_argument("--states", nargs="+", type=str, default=["open", "closed"])
        parser.add_argument(
            "--content_types", nargs="+", type=str, default=["issues", "prs"]
        )
        parser.add_argument("--verbose", type=str, default=False)
        args = parser.parse_args()

        if args.function == "scrape_gh":
            scrape_gh(
                states=args.states,
                content_types=args.content_types,
                verbose=args.verbose,
            )
        elif args.function == "create_df":
            create_df(states=args.states, content_types=args.content_types)
        elif args.function == "create_vector_db":
            create_vector_db(states=args.states, content_types=args.content_types)
        elif args.function == "run_all":
            run_all(
                states=args.states,
                content_types=args.content_types,
                verbose=args.verbose,
            )
        else:
            raise ValueError(f"Unknown function: {args.function}")
