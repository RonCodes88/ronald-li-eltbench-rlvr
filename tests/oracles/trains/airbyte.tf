provider "airbyte" {
  server_url = "$server_url"
  username   = "$username"
  password   = "$password"
}

resource "airbyte_source_custom" "custom_api" {
  name          = "Custom API Source"
  workspace_id  = "$workspace_id"
  definition_id = "$custom_api_definition_id"
  configuration = jsonencode({})
}

resource "airbyte_source_file" "train" {
  name          = "CSV File train"
  workspace_id  = "$workspace_id"
  definition_id = "$files_definition_id"
  configuration = {
    dataset_name = "train"
    format       = "csv"
    provider = {
      https_public_web = {}
    }
    url = "$train_url"
  }
}

resource "airbyte_destination_snowflake" "snowflake" {
  name          = "Snowflake Destination"
  workspace_id  = "$workspace_id"
  definition_id = "$snowflake_definition_id"
  configuration = {
    database  = "$database"
    host      = "$account.snowflakecomputing.com"
    role      = "$role"
    schema    = "$schema"
    username  = "$sf_username"
    warehouse = "$warehouse"
    credentials = {
      username_and_password = {
        password = "$sf_password"
      }
    }
  }
}

resource "airbyte_connection" "custom_api_to_snowflake" {
  name                 = "Custom API to Snowflake"
  source_id            = airbyte_source_custom.custom_api.source_id
  destination_id       = airbyte_destination_snowflake.snowflake.destination_id
  namespace_definition = "destination"
  configurations = {
    streams = [
      {
        name      = "cars"
        sync_mode = "full_refresh_append"
      }
    ]
  }
}

resource "airbyte_connection" "file_to_snowflake" {
  name                 = "File to Snowflake"
  source_id            = airbyte_source_file.train.source_id
  destination_id       = airbyte_destination_snowflake.snowflake.destination_id
  namespace_definition = "destination"
  configurations = {
    streams = [
      {
        name      = "train"
        sync_mode = "full_refresh_append"
      }
    ]
  }
}
